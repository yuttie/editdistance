use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pymodule]
fn editdistance(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(d, m)?)?;
    m.add_function(wrap_pyfunction!(nops, m)?)?;
    Ok(())
}

mod rs {
    use std::vec::Vec;
    use ndarray::Array2;

    /// Compute a DP table.
    pub fn dp(s: &str, t: &str, ins_cost: usize, del_cost: usize, sub_cost: usize) -> Array2<usize> {
        let s = s.chars().collect::<Vec<char>>();
        let t = t.chars().collect::<Vec<char>>();
        let m = s.len();
        let n = t.len();

        let mut d = Array2::<usize>::zeros((m + 1, n + 1));
        for i in 0..m + 1 {
            d[[i, 0]] = i;
        }
        for j in 0..n + 1 {
            d[[0, j]] = j;
        }
        for i in 1..m + 1 {
            for j in 1..n + 1 {
                d[[i, j]] = (d[[i - 1, j    ]] + del_cost)
                    .min(d[[i,     j - 1]] + ins_cost)
                    .min(d[[i - 1, j - 1]] + if s[i - 1] == t[j - 1] { 0 } else { sub_cost });
                }
        }

        d
    }
}

/// Count the number of operations for each type.
#[pyfunction]
fn d(s: &str, t: &str, ins_cost: usize, del_cost: usize, sub_cost: usize) -> PyResult<usize> {
    let d = rs::dp(s, t, ins_cost, del_cost, sub_cost);
    let shape = d.shape();
    let m = shape[0] - 1;
    let n = shape[1] - 1;

    Ok(d[[m, n]])
}

/// Count the number of operations for each type.
#[pyfunction]
fn nops(s: &str, t: &str, ins_cost: usize, del_cost: usize, sub_cost: usize) -> PyResult<(usize, usize, usize)> {
    let d = rs::dp(s, t, ins_cost, del_cost, sub_cost);
    let mut nins = 0;
    let mut ndel = 0;
    let mut nsub = 0;
    let shape = d.shape();
    let mut i = shape[0] - 1;
    let mut j = shape[1] - 1;
    while !(i == 0 && j == 0) {
        if d[[i - 1, j - 1]] <= d[[i, j - 1]] {
            if d[[i - 1, j - 1]] <= d[[i - 1, j]] {
                nsub += if d[[i -1, j - 1]] != d[[i, j]] { 1 } else { 0 };
                i -= 1;
                j -= 1;
            }
            else {
                ndel += 1;
                i -= 1;
            }
        }
        else {
            if d[[i, j - 1]] <= d[[i - 1, j]] {
                nins += 1;
                j -= 1;
            }
            else {
                ndel += 1;
                i -= 1;
            }
        }
    }

    Ok((nins, ndel, nsub))
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
    fn test_d() {
        assert_eq!(d("sunday", "saturday", 1, 1, 1).unwrap(), 3);
        assert_eq!(d("sitting", "kitten", 1, 1, 1).unwrap(), 3);
    }

    #[test]
    fn test_nops() {
        assert_eq!(nops("sunday", "saturday", 1, 1, 1).unwrap(), (2, 0, 1));
        assert_eq!(nops("sitting", "kitten", 1, 1, 1).unwrap(), (0, 1, 2));
    }
}
