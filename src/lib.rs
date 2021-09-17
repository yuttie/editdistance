use std::vec::Vec;
use std::collections::HashSet;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::exceptions::PyValueError;
use ndarray::Ix2;
use numpy::{PyArray, ToPyArray};

#[pymodule]
fn lcs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dp, m)?)?;
    m.add_function(wrap_pyfunction!(collect, m)?)?;
    m.add_function(wrap_pyfunction!(len, m)?)?;

    Ok(())
}

#[pyfunction]
fn dp<'py>(py: Python<'py>, s: Vec<usize>, t: Vec<usize>) -> PyResult<&'py PyArray<u32, Ix2>> {
    match py.allow_threads(|| rs::dp(&s, &t)) {
        Some(table) => Ok(table.to_pyarray(py)),
        None => Err(PyValueError::new_err("Empty sequence is not allowed.")),
    }
}

#[pyfunction]
fn collect<'py>(py: Python<'py>, table: &'py PyArray<u32, Ix2>, s: Vec<usize>, t: Vec<usize>) -> Vec<Vec<usize>> {
    let table = table.readonly();
    let table = table.as_array();
    py.allow_threads(|| rs::collect(&table, &s, &t).into_iter().collect())
}

#[pyfunction]
fn len<'py>(py: Python<'py>, s: Vec<usize>, t: Vec<usize>) -> PyResult<u32> {
    py.allow_threads(|| rs::len(&s, &t)).ok_or(PyValueError::new_err("Empty sequence is not allowed."))
}

mod rs {
    use std::cmp::max;
    use std::vec::Vec;
    use std::hash::Hash;
    use std::collections::HashSet;
    use ndarray::{ArrayBase, Array2, Ix2, Data, RawData};

    /// Compute a DP table.
    pub fn dp<'a, T: Eq>(s: &[T], t: &[T]) -> Option<Array2<u32>> {
        let n = s.len();
        let m = t.len();

        if n == 0 || m == 0 {
            return None;
        }

        let mut table = Array2::<u32>::zeros((n + 1, m + 1));

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

        Some(table)
    }

    /// Collect all the longest common subsequences from a given DP table.
    pub fn collect<'a, T: Eq + Hash + Clone, S: Sync + Data + RawData<Elem = u32>>(table: &ArrayBase<S, Ix2>, s: &[T], t: &[T]) -> HashSet<Vec<T>> {
        fn backtrack<'a, T: Eq + Hash + Clone, S: Sync + Data + RawData<Elem = u32>>(table: &ArrayBase<S, Ix2>, s: &[T], t: &[T], i: usize, j: usize) -> HashSet<Vec<T>> {
            if i == 0 || j == 0 {
                IntoIterator::into_iter([vec![]]).collect()
            }
            else if s[i - 1] == t[j - 1] {
                backtrack(table, s, t, i - 1, j - 1).into_iter().map(|mut subseq| {
                    subseq.push(s[i - 1].clone());
                    subseq
                }).collect()
            }
            else {
                let mut res = HashSet::new();
                if table[[i, j - 1]] >= table[[i - 1, j]] {
                    res.extend(backtrack(table, s, t, i, j - 1).into_iter());
                }
                if table[[i - 1, j]] >= table[[i, j - 1]] {
                    res.extend(backtrack(table, s, t, i - 1, j).into_iter());
                }
                res
            }
        }

        let n = s.len();
        let m = t.len();
        backtrack(table, s, t, n, m)
    }

    /// Compute the length of longest common subsequence of two given sequences.
    pub fn len<'a, T: Eq>(s: &[T], t: &[T]) -> Option<u32> {
        let n = s.len();
        let m = t.len();
        dp(s, t).map(|table| table[[n, m]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array};

    #[test]
    fn test_dp() {
        let s = vec![5, 2, 1, 6, 0, 3, 7];
        let t = vec![2, 7, 1, 0, 4, 5, 3];
        assert_eq!(rs::dp(&s, &t), Some(array![
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 2, 2, 2, 2, 2],
            [0, 1, 1, 2, 2, 2, 2, 2],
            [0, 1, 1, 2, 3, 3, 3, 3],
            [0, 1, 1, 2, 3, 3, 3, 4],
            [0, 1, 2, 2, 3, 3, 3, 4],
        ]));
    }

    #[test]
    fn test_dp_with_empty_sequence() {
        let s = vec![];
        let t = vec![0, 1, 2, 3, 4];
        assert_eq!(rs::dp(&s, &t), None);

        let s = vec![0, 1, 2, 3, 4];
        let t = vec![];
        assert_eq!(rs::dp(&s, &t), None);
    }

    #[test]
    fn test_collect_equal_sequences() {
        let s = vec![0, 1, 2, 3, 4];
        let t = vec![0, 1, 2, 3, 4];
        let table = rs::dp(&s, &t);
        assert_eq!(table.is_some(), true);
        let table = table.unwrap();
        assert_eq!(rs::collect(&table, &s, &t), [
            vec![0, 1, 2, 3, 4],
        ].iter().cloned().collect());
    }

    #[test]
    fn test_collect_different_sequences1() {
        let s = vec![5, 2, 1, 6, 0, 3, 7];
        let t = vec![2, 7, 1, 0, 4, 5, 3];
        let table = rs::dp(&s, &t);
        assert_eq!(table.is_some(), true);
        let table = table.unwrap();
        assert_eq!(rs::collect(&table, &s, &t), [
            vec![2, 1, 0, 3],
        ].iter().cloned().collect());
    }

    #[test]
    fn test_collect_different_sequences2() {
        let s = vec![10, 0, 11, 12, 1, 2, 13, 14, 3, 15, 4, 16, 5, 17];
        let t = vec![0, 20, 1, 2, 3, 21, 22, 23, 4, 24, 5];
        let table = rs::dp(&s, &t);
        assert_eq!(table.is_some(), true);
        let table = table.unwrap();
        assert_eq!(rs::collect(&table, &s, &t), [
            vec![0, 1, 2, 3, 4, 5],
        ].iter().cloned().collect());
    }

    #[test]
    fn test_len_equal_sequences() {
        let s = vec![0, 1, 2, 3, 4];
        let t = vec![0, 1, 2, 3, 4];
        assert_eq!(rs::len(&s, &t), Some(5));
    }

    #[test]
    fn test_len_different_sequences1() {
        let s = vec![5, 2, 1, 6, 0, 3, 7];
        let t = vec![2, 7, 1, 0, 4, 5, 3];
        assert_eq!(rs::len(&s, &t), Some(4));
    }

    #[test]
    fn test_len_different_sequences2() {
        let s = vec![10, 0, 11, 12, 1, 2, 13, 14, 3, 15, 4, 16, 5, 17];
        let t = vec![0, 20, 1, 2, 3, 21, 22, 23, 4, 24, 5];
        assert_eq!(rs::len(&s, &t), Some(6));
    }
}
