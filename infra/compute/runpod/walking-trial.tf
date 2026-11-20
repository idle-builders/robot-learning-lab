variable "walking_trial_enabled" {
  type        = bool
  default     = false
  description = "Create the temporary, scale-to-zero R3 simulation/training endpoint."
}

variable "walking_trial_timeout_ms" {
  type        = number
  default     = 1800000
  description = "Provider execution deadline; reviewed continuations may use 60, 75, 115, 120 or 140 minutes."
  validation {
    condition     = contains([1800000, 3600000, 4500000, 6900000, 7200000, 8400000], var.walking_trial_timeout_ms)
    error_message = "Use the bounded 30-minute, 60-minute, 75-minute, 115-minute, 120-minute or 140-minute execution deadline."
  }
}

variable "walking_trial_ampere24_fallback" {
  type        = bool
  default     = false
  description = "Also allow the 24 GB L4/A5000/3090 pool when reviewed for a bounded trial."
}

variable "walking_trial_large_gpu" {
  type        = bool
  default     = false
  description = "Use the large pools for a reviewed trial: 48 GB L40/L40S/6000 Ada first (most FP32 for R3 physics), then H100 80 GB, A100 80 GB last."
}

variable "walking_trial_workers_max" {
  type        = number
  default     = 1
  description = "Worker cap; a reviewed batch of independent jobs may run up to four side by side, then returns to one."
  validation {
    condition     = contains([1, 2, 3, 4], var.walking_trial_workers_max)
    error_message = "Use one worker, or at most four for a reviewed batch."
  }
}

resource "runpod_serverless_endpoint" "walking_trial" {
  count         = var.walking_trial_enabled ? 1 : 0
  name          = "agent-cove50-walking-trial"
  image         = "runpod/flash@sha256:25606809ebed07a94b2e75eee7baa9c10515cf75ef6d09bb33b2695a30db7f20"
  endpoint_type = "QUEUE"
  gpu_pools = (var.walking_trial_large_gpu ? ["ADA_48_PRO", "ADA_80_PRO", "AMPERE_80"] :
  var.walking_trial_ampere24_fallback ? ["ADA_24", "AMPERE_24"] : ["ADA_24"])
  gpu_count     = 1
  disk          = 100
  workers_min   = 0
  workers_max   = var.walking_trial_workers_max
  idle_timeout  = 5
  timeout       = var.walking_trial_timeout_ms
  flashboot     = "FLASHBOOT"
  scaling_type  = "QUEUE_DELAY"
  scaling_value = 4
  env = {
    # This trial computes physics only; replay videos are rendered locally from native state.
    MUJOCO_GL                  = "disable"
    NVIDIA_DRIVER_CAPABILITIES = "compute,utility,graphics"
    RUNPOD_API_KEY             = ""
    OMP_NUM_THREADS            = "4"
  }
}

output "walking_trial_id" {
  value = try(runpod_serverless_endpoint.walking_trial[0].id, null)
}
