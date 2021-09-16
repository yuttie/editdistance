use std::vec::Vec;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::exceptions::PyValueError;
use ndarray::Ix2;
use numpy::{PyArray, ToPyArray};

mod rs {
    use std::cmp::max;
    use std::vec::Vec;
    use ndarray::{ArrayBase, Array2, Ix2, Data, RawData};
    use ndarray::parallel::prelude::par_azip;

    /// Compute a DP table.
    pub fn dp<'a>(s: &[usize], t: &[usize]) -> Option<Array2<u32>> {
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
    pub fn collect<'a, S: Sync + Data + RawData<Elem = u32>>(table: &ArrayBase<S, Ix2>, s: &[usize], t: &[usize]) -> Vec<Vec<usize>> {
        fn backtrack<'a, S: Sync + Data + RawData<Elem = u32>>(table: &ArrayBase<S, Ix2>, s: &[usize], t: &[usize], i: usize, j: usize) -> Vec<Vec<usize>> {
            if i == 0 || j == 0 {
                vec![vec![]]
            }
            else if s[i - 1] == t[j - 1] {
                backtrack(table, s, t, i - 1, j - 1).iter().map(|x| {
                    let mut x = x.clone();
                    x.push(s[i - 1]);
                    x
                }).collect()
            }
            else {
                let mut res = Vec::new();
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
    pub fn len<'a>(s: &[usize], t: &[usize]) -> Option<u32> {
        let n = s.len();
        let m = t.len();
        dp(s, t).map(|table| table[[n, m]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

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
        assert_eq!(rs::collect(&table, &s, &t), vec![
            vec![0, 1, 2, 3, 4],
        ]);
    }

    #[test]
    fn test_collect_different_sequences() {
        let s = vec![5, 2, 1, 6, 0, 3, 7];
        let t = vec![2, 7, 1, 0, 4, 5, 3];
        let table = rs::dp(&s, &t);
        assert_eq!(table.is_some(), true);
        let table = table.unwrap();
        assert_eq!(rs::collect(&table, &s, &t), vec![
            vec![2, 1, 0, 3],
        ]);
    }

    #[test]
    fn test_len_equal_sequences() {
        let s = vec![0, 1, 2, 3, 4];
        let t = vec![0, 1, 2, 3, 4];
        assert_eq!(rs::len(&s, &t), Some(5));
    }

    #[test]
    fn test_len_different_sequences() {
        let s = vec![5, 2, 1, 6, 0, 3, 7];
        let t = vec![2, 7, 1, 0, 4, 5, 3];
        assert_eq!(rs::len(&s, &t), Some(4));
    }
}
