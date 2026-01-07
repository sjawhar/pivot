"""Pipeline executor package - runs stages with greedy parallel execution.

Public API:
    run() - Execute pipeline stages
    ExecutionSummary - Result type for stage execution
    ChangeCheckResult - Change detection result type
    check_stage_changed - Check if stage needs to run
    WorkerStageInfo - Stage info passed to workers
    execute_stage - Worker function
    hash_dependencies - Hash dependency files
    hash_file - Hash a single file
"""

from pivot.executor import worker as worker
from pivot.executor.core import ChangeCheckResult as ChangeCheckResult
from pivot.executor.core import ExecutionSummary as ExecutionSummary
from pivot.executor.core import check_stage_changed as check_stage_changed
from pivot.executor.core import run as run
from pivot.executor.worker import WorkerStageInfo as WorkerStageInfo
from pivot.executor.worker import execute_stage as execute_stage
from pivot.executor.worker import hash_dependencies as hash_dependencies
from pivot.executor.worker import hash_file as hash_file

# Backwards compatibility alias
execute_stage_worker = execute_stage
