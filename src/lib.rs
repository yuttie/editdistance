use std::vec::Vec;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use ndarray::Dim;
use numpy::{PyArray, ToPyArray};

#[pymodule]
fn editdistance(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dist, m)?)?;
    m.add_function(wrap_pyfunction!(nops, m)?)?;
    m.add_function(wrap_pyfunction!(pdist, m)?)?;

    Ok(())
}

#[pyfunction]
fn dist(py: Python, s: &str, t: &str, ins_cost: u32, del_cost: u32, sub_cost: u32) -> u32 {
    py.allow_threads(|| rs::dist(s, t, ins_cost, del_cost, sub_cost))
}

#[pyfunction]
fn nops(py: Python, s: &str, t: &str, ins_cost: u32, del_cost: u32, sub_cost: u32) -> (u32, u32, u32) {
    py.allow_threads(|| rs::nops(s, t, ins_cost, del_cost, sub_cost))
}

#[pyfunction]
fn pdist<'py>(py: Python<'py>, xs: Vec<&str>, ins_cost: u32, del_cost: u32, sub_cost: u32) -> &'py PyArray<u32, Dim<[usize; 2]>> {
    py.allow_threads(|| rs::pdist(&xs, ins_cost, del_cost, sub_cost)).to_pyarray(py)
}

mod rs {
    use std::convert::TryInto;
    use std::vec::Vec;
    use ndarray::Array2;
    use ndarray::parallel::prelude::par_azip;

    /// Compute a DP table.
    pub fn dp(s: &str, t: &str, ins_cost: u32, del_cost: u32, sub_cost: u32) -> Array2<u32> {
        let s = s.chars().collect::<Vec<char>>();
        let t = t.chars().collect::<Vec<char>>();
        let m = s.len();
        let n = t.len();

        let mut table = Array2::<u32>::zeros((m + 1, n + 1));
        for i in 0..m + 1 {
            table[[i, 0]] = i.try_into().unwrap();
        }
        for j in 0..n + 1 {
            table[[0, j]] = j.try_into().unwrap();
        }
        if m > 0 && n > 0 {
            for i in 1..m + 1 {
                for j in 1..n + 1 {
                    table[[i, j]] = (table[[i - 1, j    ]] + del_cost)
                        .min(table[[i,     j - 1]] + ins_cost)
                        .min(table[[i - 1, j - 1]] + if s[i - 1] == t[j - 1] { 0 } else { sub_cost });
                }
            }
        }

        table
    }

    /// Count the number of operations for each type.
    pub fn dist(s: &str, t: &str, ins_cost: u32, del_cost: u32, sub_cost: u32) -> u32 {
        let table = dp(s, t, ins_cost, del_cost, sub_cost);
        let shape = table.shape();
        let m = shape[0] - 1;
        let n = shape[1] - 1;

        table[[m, n]]
    }

    /// Count the number of operations for each type.
    pub fn nops(s: &str, t: &str, ins_cost: u32, del_cost: u32, sub_cost: u32) -> (u32, u32, u32) {
        let table = dp(s, t, ins_cost, del_cost, sub_cost);
        let mut nins = 0;
        let mut ndel = 0;
        let mut nsub = 0;
        let shape = table.shape();
        let mut i = shape[0] - 1;
        let mut j = shape[1] - 1;
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
    pub fn pdist<T: AsRef<str> + Sync>(xs: &[T], ins_cost: u32, del_cost: u32, sub_cost: u32) -> Array2<u32> {
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

    #[test]
    fn test_dp() {
        assert_eq!(rs::dp("sunday", "saturday", 1, 1, 1), array![
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [1, 0, 1, 2, 3, 4, 5, 6, 7],
            [2, 1, 1, 2, 2, 3, 4, 5, 6],
            [3, 2, 2, 2, 3, 3, 4, 5, 6],
            [4, 3, 3, 3, 3, 4, 3, 4, 5],
            [5, 4, 3, 4, 4, 4, 4, 3, 4],
            [6, 5, 4, 4, 5, 5, 5, 4, 3],
        ]);
        assert_eq!(rs::dp("sitting", "kitten", 1, 1, 1), array![
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
        assert_eq!(rs::dist("sunday", "saturday", 1, 1, 1), 3);
        assert_eq!(rs::dist("sitting", "kitten", 1, 1, 1), 3);
        assert_eq!(rs::dist("", "empty", 1, 1, 1), 5);
        assert_eq!(rs::dist("empty", "", 1, 1, 1), 5);
    }

    #[test]
    fn test_nops() {
        assert_eq!(rs::nops("sunday", "saturday", 1, 1, 1), (2, 0, 1));
        assert_eq!(rs::nops("sitting", "kitten", 1, 1, 1), (0, 1, 2));
        assert_eq!(rs::nops("", "empty", 1, 1, 1), (5, 0, 0));
        assert_eq!(rs::nops("empty", "", 1, 1, 1), (0, 5, 0));
    }

    #[test]
    fn test_pdist() {
        assert_eq!(rs::pdist(&["sunday", "saturday", "sitting", "kitten", ""], 1, 1, 1), array![
            [0, 3, 6, 6, 6],
            [3, 0, 6, 7, 8],
            [6, 6, 0, 3, 7],
            [6, 7, 3, 0, 6],
            [6, 8, 7, 6, 0],
        ]);
    }
}
