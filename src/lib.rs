use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod lcs;
mod levenshtein;

#[pymodule]
fn editdistance(py: Python, m: &PyModule) -> PyResult<()> {
    let lcs_module = PyModule::new(py, "lcs")?;
    lcs_module.add_function(wrap_pyfunction!(lcs::dp, m)?)?;
    lcs_module.add_function(wrap_pyfunction!(lcs::collect, m)?)?;
    lcs_module.add_function(wrap_pyfunction!(lcs::len, m)?)?;

    let levenshtein_module = PyModule::new(py, "levenshtein")?;
    levenshtein_module.add_function(wrap_pyfunction!(levenshtein::dist, m)?)?;
    levenshtein_module.add_function(wrap_pyfunction!(levenshtein::nops, m)?)?;
    levenshtein_module.add_function(wrap_pyfunction!(levenshtein::pdist, m)?)?;

    m.add_submodule(lcs_module)?;
    m.add_submodule(levenshtein_module)?;

    Ok(())
}
