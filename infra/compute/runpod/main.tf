terraform {
  required_version = ">= 1.6, < 2.0"
  required_providers {
    runpod = {
      source  = "runpod/runpod"
      version = "2.0.0-rc.3"
    }
  }
}

# The launcher supplies RUNPOD_API_KEY from the local authenticated CLI profile.
provider "runpod" {}

# New account bootstrap. GPU jobs remain explicit, bounded experiment launches.
# Adapted from Runpod's pinned v2 network-volume module.
resource "runpod_network_volume" "transport" {
  name        = "agent-robot-lab-transport"
  size        = 10
  data_center = "EU-RO-1"
  type        = "STANDARD"

  lifecycle {
    prevent_destroy = true
  }
}

resource "runpod_template" "walking" {
  name          = "agent-cove50-walking"
  image         = "runpod/pytorch@sha256:3e874356857adfa3e8faa3fd913b65bd127f77a0fe2e489513e7775e1c1e16b1"
  disk          = 100
  public        = false
  serverless    = false
  start_ssh     = true
  start_jupyter = false
  ports         = ["22/tcp"]
  env = {
    MUJOCO_GL                  = "egl"
    NVIDIA_DRIVER_CAPABILITIES = "compute,utility,graphics"
    ROBOT_LAB_VOLUME_ID        = runpod_network_volume.transport.id
  }
}

output "transport" {
  value = {
    id          = runpod_network_volume.transport.id
    data_center = runpod_network_volume.transport.data_center
    size_gb     = runpod_network_volume.transport.size
    s3_endpoint = "https://s3api-eu-ro-1.runpod.io/"
  }
}

output "walking_template_id" {
  value = runpod_template.walking.id
}
