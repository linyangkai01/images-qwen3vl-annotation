param(
    [string]$Config = "examples/mock_project/config/project.json",
    [switch]$ExecuteTrainer
)

$args = @("-m", "traffic_cv_closed_loop", "train", "--config", $Config)
if ($ExecuteTrainer) {
    $args += "--execute-trainer"
}

python @args

