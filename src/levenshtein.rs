use std::vec::Vec;
use pyo3::prelude::*;
use pyo3::types::PyString;
use ndarray::Ix2;
use numpy::{PyArray, ToPyArray};

#[pyfunction]
pub fn dist(py: Python, s: PyObject, t: PyObject, ins_cost: u32, del_cost: u32, sub_cost: u32) -> PyResult<u32> {
    let s: Vec<u32> = normalize_input(py, s)?;
    let t: Vec<u32> = normalize_input(py, t)?;
    py.allow_threads(|| Ok(rs::dist(&s, &t, ins_cost, del_cost, sub_cost)))
}

#[pyfunction]
pub fn nops(py: Python, s: PyObject, t: PyObject, ins_cost: u32, del_cost: u32, sub_cost: u32) -> PyResult<(u32, u32, u32)> {
    let s: Vec<u32> = normalize_input(py, s)?;
    let t: Vec<u32> = normalize_input(py, t)?;
    py.allow_threads(|| Ok(rs::nops(&s, &t, ins_cost, del_cost, sub_cost)))
}

#[pyfunction]
pub fn pdist<'py>(py: Python<'py>, xs: Vec<PyObject>, ins_cost: u32, del_cost: u32, sub_cost: u32) -> PyResult<&'py PyArray<u32, Ix2>> {
    let xs: Vec<Vec<u32>> = xs.into_iter().map(|x| normalize_input(py, x)).collect::<Result<_, _>>()?;
    Ok(py.allow_threads(|| rs::pdist(&xs, ins_cost, del_cost, sub_cost)).to_pyarray(py))
}

fn normalize_input<'py>(py: Python<'py>, s: PyObject) -> PyResult<Vec<u32>> {
    let s: &PyAny = s.as_ref(py);
    if s.is_instance_of::<PyString>().unwrap() {
        let s: &str = s.extract()?;
        Ok(s.chars().map(|c| u32::from(c)).collect())
    }
    else {
        Ok(s.extract()?)
    }
}

mod rs {
    use std::convert::TryInto;
    use std::vec::Vec;
    use ndarray::Array2;
    use ndarray::parallel::prelude::par_azip;

    /// Compute a DP table.
    pub fn dp<'a, T: Eq>(s: &[T], t: &[T], ins_cost: u32, del_cost: u32, sub_cost: u32) -> Array2<u32> {
        let n = s.len();
        let m = t.len();

        let mut table = Array2::<u32>::zeros((n + 1, m + 1));
        for i in 0..=n {
            table[[i, 0]] = i.try_into().unwrap();
        }
        for j in 0..=m {
            table[[0, j]] = j.try_into().unwrap();
        }
        if n > 0 && m > 0 {
            for i in 1..=n {
                for j in 1..=m {
                    table[[i, j]] = (table[[i - 1, j    ]] + del_cost)
                        .min(table[[i,     j - 1]] + ins_cost)
                        .min(table[[i - 1, j - 1]] + if s[i - 1] == t[j - 1] { 0 } else { sub_cost });
                }
            }
        }

        table
    }

    /// Compute the Levenshtein distance of two given sequences.
    pub fn dist<'a, T: Eq>(s: &[T], t: &[T], ins_cost: u32, del_cost: u32, sub_cost: u32) -> u32 {
        let table = dp(s, t, ins_cost, del_cost, sub_cost);
        let n = s.len();
        let m = t.len();

        table[[n, m]]
    }

    /// Count the number of operations for each type.
    pub fn nops<'a, T: Eq>(s: &[T], t: &[T], ins_cost: u32, del_cost: u32, sub_cost: u32) -> (u32, u32, u32) {
        let table = dp(s, t, ins_cost, del_cost, sub_cost);
        let mut nins = 0;
        let mut ndel = 0;
        let mut nsub = 0;
        let mut i = s.len();
        let mut j = t.len();
        while !(i == 0 || j == 0) {
            if table[[i - 1, j - 1]] <= table[[i, j - 1]] {
                if table[[i - 1, j - 1]] <= table[[i - 1, j]] {
                    nsub += if table[[i -1, j - 1]] != table[[i, j]] { 1 } else { 0 };
                    i -= 1;
                    j -= 1;
                }
                else {
                    ndel += 1;
                    i -= 1;
                }
            }
            else {
                if table[[i, j - 1]] <= table[[i - 1, j]] {
                    nins += 1;
                    j -= 1;
                }
                else {
                    ndel += 1;
                    i -= 1;
                }
            }
        }
        while i > 0 {
            ndel += 1;
            i -= 1;
        }
        while j > 0 {
            nins += 1;
            j -= 1;
        }

        (nins, ndel, nsub)
    }

    /// Compute pairwise distances between strings.
    pub fn pdist<T: Eq + Sync>(xs: &[Vec<T>], ins_cost: u32, del_cost: u32, sub_cost: u32) -> Array2<u32> {
        let n = xs.len();
        let mut dmat = Array2::<u32>::zeros((n, n));
        let indices = ndarray::indices_of(&dmat);
        par_azip!((dmat_ij in &mut dmat, (i, j) in indices) {
            *dmat_ij = dist(xs[i].as_ref(), xs[j].as_ref(), ins_cost, del_cost, sub_cost);
        });
        dmat
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn to_vec(s: &str) -> Vec<u32> {
        s.chars().map(|c| u32::from(c)).collect()
    }

    #[test]
    fn test_dp() {
        assert_eq!(rs::dp(&to_vec("sunday"), &to_vec("saturday"), 1, 1, 1), array![
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [1, 0, 1, 2, 3, 4, 5, 6, 7],
            [2, 1, 1, 2, 2, 3, 4, 5, 6],
            [3, 2, 2, 2, 3, 3, 4, 5, 6],
            [4, 3, 3, 3, 3, 4, 3, 4, 5],
            [5, 4, 3, 4, 4, 4, 4, 3, 4],
            [6, 5, 4, 4, 5, 5, 5, 4, 3],
        ]);
        assert_eq!(rs::dp(&to_vec("sitting"), &to_vec("kitten"), 1, 1, 1), array![
            [0, 1, 2, 3, 4, 5, 6],
            [1, 1, 2, 3, 4, 5, 6],
            [2, 2, 1, 2, 3, 4, 5],
            [3, 3, 2, 1, 2, 3, 4],
            [4, 4, 3, 2, 1, 2, 3],
            [5, 5, 4, 3, 2, 2, 3],
            [6, 6, 5, 4, 3, 3, 2],
            [7, 7, 6, 5, 4, 4, 3],
        ]);
    }

    #[test]
    fn test_dist() {
        assert_eq!(rs::dist(&to_vec("sunday"), &to_vec("saturday"), 1, 1, 1), 3);
        assert_eq!(rs::dist(&to_vec("sitting"), &to_vec("kitten"), 1, 1, 1), 3);
        assert_eq!(rs::dist(&to_vec(""), &to_vec("empty"), 1, 1, 1), 5);
        assert_eq!(rs::dist(&to_vec("empty"), &to_vec(""), 1, 1, 1), 5);
    }

    #[test]
    fn test_nops() {
        assert_eq!(rs::nops(&to_vec("sunday"), &to_vec("saturday"), 1, 1, 1), (2, 0, 1));
        assert_eq!(rs::nops(&to_vec("sitting"), &to_vec("kitten"), 1, 1, 1), (0, 1, 2));
        assert_eq!(rs::nops(&to_vec(""), &to_vec("empty"), 1, 1, 1), (5, 0, 0));
        assert_eq!(rs::nops(&to_vec("empty"), &to_vec(""), 1, 1, 1), (0, 5, 0));
    }

    #[test]
    fn test_pdist() {
        assert_eq!(rs::pdist(&[to_vec("sunday"), to_vec("saturday"), to_vec("sitting"), to_vec("kitten"), to_vec("")], 1, 1, 1), array![
            [0, 3, 6, 6, 6],
            [3, 0, 6, 7, 8],
            [6, 6, 0, 3, 7],
            [6, 7, 3, 0, 6],
            [6, 8, 7, 6, 0],
        ]);
    }
}
