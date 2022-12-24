use std::vec::Vec;
use pyo3::prelude::*;
use pyo3::types::PyString;
use ndarray::Ix2;
use numpy::{PyArray, ToPyArray};

#[pyfunction]
pub fn dp<'py>(py: Python<'py>, s: PyObject, t: PyObject) -> PyResult<&'py PyArray<u32, Ix2>> {
    let s: Vec<u32> = normalize_input(py, s)?;
    let t: Vec<u32> = normalize_input(py, t)?;
    Ok(py.allow_threads(|| rs::dp(&s, &t)).to_pyarray(py))
}

#[pyfunction]
pub fn collect<'py>(py: Python<'py>, s: PyObject, t: PyObject) -> PyResult<Vec<Vec<u32>>> {
    let s: Vec<u32> = normalize_input(py, s)?;
    let t: Vec<u32> = normalize_input(py, t)?;
    py.allow_threads(|| Ok(rs::collect(&s, &t).into_iter().collect()))
}

#[pyfunction]
pub fn len<'py>(py: Python<'py>, s: PyObject, t: PyObject) -> PyResult<u32> {
    let s: Vec<u32> = normalize_input(py, s)?;
    let t: Vec<u32> = normalize_input(py, t)?;
    py.allow_threads(|| Ok(rs::len(&s, &t)))
}

#[pyfunction]
pub fn dist<'py>(py: Python<'py>, s: PyObject, t: PyObject) -> PyResult<u32> {
    let s: Vec<u32> = normalize_input(py, s)?;
    let t: Vec<u32> = normalize_input(py, t)?;
    py.allow_threads(|| Ok(rs::dist(&s, &t)))
}

#[pyfunction]
pub fn sim<'py>(py: Python<'py>, s: PyObject, t: PyObject) -> PyResult<f64> {
    let s: Vec<u32> = normalize_input(py, s)?;
    let t: Vec<u32> = normalize_input(py, t)?;
    py.allow_threads(|| Ok(rs::sim(&s, &t)))
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
    use std::cmp::max;
    use std::vec::Vec;
    use std::hash::Hash;
    use std::collections::{HashMap, HashSet};
    use ndarray::Array2;

    /// Compute a DP table.
    pub fn dp<'a, T: Eq>(s: &[T], t: &[T]) -> Array2<u32> {
        let n = s.len();
        let m = t.len();

        let mut table = Array2::<u32>::zeros((n + 1, m + 1));

        if n > 0 && m > 0 {
            for i in 1..=n {
                for j in 1..=m {
                    table[[i, j]] = if s[i - 1] == t[j - 1] {
                        table[[i - 1, j - 1]] + 1
                    }
                    else {
                        max(table[[i, j - 1]], table[[i - 1, j]])
                    };
                }
            }
        }

        table
    }

    /// Collect all the longest common subsequences.
    pub fn collect<'a, T: Eq + Hash + Clone + std::fmt::Debug>(s: &[T], t: &[T]) -> HashSet<Vec<T>> {
        let n = s.len();
        let m = t.len();

        if n == 0 || m == 0 {
            return HashSet::new();
        }

        let mut table: HashMap<(usize, usize), HashSet<Vec<T>>> = HashMap::new();
        table.insert((0, 0), IntoIterator::into_iter([vec![]]).collect());
        for i in 1..=n {
            table.insert((i, 0), IntoIterator::into_iter([vec![]]).collect());
        }
        for j in 1..=m {
            table.insert((0, j), IntoIterator::into_iter([vec![]]).collect());
        }

        for i in 1..=n {
            for j in 1..=m {
                let lcs_ij = if s[i - 1] == t[j - 1] {
                    table[&(i - 1, j - 1)].iter().cloned().map(|mut subseq| {
                        subseq.push(s[i - 1].clone());
                        subseq
                    }).collect()
                }
                else {
                    let mut res = HashSet::new();
                    if table[&(i, j - 1)].iter().next().unwrap().len() >= table[&(i - 1, j)].iter().next().unwrap().len() {
                        res.extend(table[&(i, j - 1)].iter().cloned());
                    }
                    if table[&(i - 1, j)].iter().next().unwrap().len() >= table[&(i, j - 1)].iter().next().unwrap().len() {
                        res.extend(table[&(i - 1, j)].iter().cloned());
                    }
                    res
                };
                table.insert((i, j), lcs_ij);
            }
        }

        table.remove(&(n, m)).unwrap()
    }

    /// Compute the length of longest common subsequence of two given sequences.
    pub fn len<'a, T: Eq>(s: &[T], t: &[T]) -> u32 {
        let n = s.len();
        let m = t.len();
        let table = dp(s, t);
        table[[n, m]]
    }

    /// Compute the LCS distance of two given sequences.
    pub fn dist<'a, T: Eq>(s: &[T], t: &[T]) -> u32 {
        let n = s.len();
        let m = t.len();
        let l = len(s, t);
        u32::try_from(n + m).unwrap() - 2 * l
    }

    /// Compute the LCS similarity of two given sequences.
    pub fn sim<'a, T: Eq>(s: &[T], t: &[T]) -> f64 {
        let n = s.len();
        let m = t.len();
        let l = len(s, t);
        if n + m == 0 {
            1.
        }
        else {
            f64::from(2 * l) / (n + m) as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dp() {
        let s = vec![5, 2, 1, 6, 0, 3, 7];
        let t = vec![2, 7, 1, 0, 4, 5, 3];
        assert_eq!(rs::dp(&s, &t), array![
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 2, 2, 2, 2, 2],
            [0, 1, 1, 2, 2, 2, 2, 2],
            [0, 1, 1, 2, 3, 3, 3, 3],
            [0, 1, 1, 2, 3, 3, 3, 4],
            [0, 1, 2, 2, 3, 3, 3, 4],
        ]);
    }

    #[test]
    fn test_dp_with_empty_sequence() {
        let s = vec![];
        let t = vec![0, 1, 2, 3, 4];
        assert_eq!(rs::dp(&s, &t), array![
            [0, 0, 0, 0, 0, 0],
        ]);

        let s = vec![0, 1, 2, 3, 4];
        let t = vec![];
        assert_eq!(rs::dp(&s, &t), array![
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
        ]);
    }

    #[test]
    fn test_collect_equal_sequences() {
        let s = vec![0, 1, 2, 3, 4];
        let t = vec![0, 1, 2, 3, 4];
        let _table = rs::dp(&s, &t);
        assert_eq!(rs::collect(&s, &t), [
            vec![0, 1, 2, 3, 4],
        ].iter().cloned().collect());
    }

    #[test]
    fn test_collect_different_sequences1() {
        let s = vec![5, 2, 1, 6, 0, 3, 7];
        let t = vec![2, 7, 1, 0, 4, 5, 3];
        let _table = rs::dp(&s, &t);
        assert_eq!(rs::collect(&s, &t), [
            vec![2, 1, 0, 3],
        ].iter().cloned().collect());
    }

    #[test]
    fn test_collect_different_sequences2() {
        let s = vec![10, 0, 11, 12, 1, 2, 13, 14, 3, 15, 4, 16, 5, 17];
        let t = vec![0, 20, 1, 2, 3, 21, 22, 23, 4, 24, 5];
        let _table = rs::dp(&s, &t);
        assert_eq!(rs::collect(&s, &t), [
            vec![0, 1, 2, 3, 4, 5],
        ].iter().cloned().collect());
    }

    #[test]
    fn test_len_equal_sequences() {
        let s = vec![0, 1, 2, 3, 4];
        let t = vec![0, 1, 2, 3, 4];
        assert_eq!(rs::len(&s, &t), 5);
    }

    #[test]
    fn test_len_different_sequences1() {
        let s = vec![5, 2, 1, 6, 0, 3, 7];
        let t = vec![2, 7, 1, 0, 4, 5, 3];
        assert_eq!(rs::len(&s, &t), 4);
    }

    #[test]
    fn test_len_different_sequences2() {
        let s = vec![10, 0, 11, 12, 1, 2, 13, 14, 3, 15, 4, 16, 5, 17];
        let t = vec![0, 20, 1, 2, 3, 21, 22, 23, 4, 24, 5];
        assert_eq!(rs::len(&s, &t), 6);
    }
}
