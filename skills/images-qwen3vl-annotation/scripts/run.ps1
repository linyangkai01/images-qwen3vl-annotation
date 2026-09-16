param(
    [string]$Config = "examples/mock_project/config/project.json"
)

python -m traffic_cv_closed_loop annotate --config $Config

