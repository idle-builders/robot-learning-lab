"""Image-derived fiducial perception and an explicit authored skill adapter.

OpenCV owns detection and pose estimation. No simulator state enters this path.
This bounded bootstrap is not general object recognition or a learned selector.
Forward/left are camera-relative. The drive adapter requires a neutral, fixed
head; arbitrary neck poses require a calibrated camera-to-base transform.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TargetEstimate:
    acquired_s: float
    marker_id: int
    forward_m: float
    left_m: float
    reprojection_error_px: float


def detect_target(rgb, camera_matrix, marker_size_m, acquired_s, marker_id=7):
    import cv2
    import numpy as np
    if marker_size_m <= 0 or not math.isfinite(acquired_s):
        raise ValueError('Positive marker size and finite acquisition time required')
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners,ids,_=cv2.aruco.ArucoDetector(dictionary).detectMarkers(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))
    if ids is None:return None
    matches=np.flatnonzero(ids.ravel()==marker_id)
    if len(matches)!=1:return None
    image_points=corners[int(matches[0])].reshape(4,2)
    h=marker_size_m/2
    object_points=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]],dtype=np.float32)
    k=np.asarray(camera_matrix,dtype=float)
    ok,rvec,tvec=cv2.solvePnP(object_points,image_points,k,None,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or not np.isfinite(tvec).all() or tvec[2,0]<=0:return None
    projected,_=cv2.projectPoints(object_points,rvec,tvec,k,None)
    error=float(np.linalg.norm(projected.reshape(4,2)-image_points,axis=1).mean())
    if error>3:return None
    # Optical right/down/forward -> neutral-head forward/left/up.
    return TargetEstimate(acquired_s,marker_id,float(tvec[2,0]),float(-tvec[0,0]),error)


@dataclass(frozen=True)
class SkillCommand:
    skill: str
    forward_m_s: float
    yaw_rad_s: float
    reason: str


def approach_skill(estimate, now_s, *, stand_off_m=0.45, max_age_s=0.15):
    if not math.isfinite(now_s):raise ValueError('Finite control time required')
    if estimate is None:return SkillCommand('hold',0,0,'target_missing')
    values=[estimate.acquired_s,estimate.forward_m,estimate.left_m,estimate.reprojection_error_px]
    if not all(math.isfinite(x) for x in values) or not 0<=now_s-estimate.acquired_s<=max_age_s:
        return SkillCommand('hold',0,0,'target_stale_or_invalid')
    if estimate.forward_m<=stand_off_m:return SkillCommand('hold',0,0,'stand_off_reached')
    yaw=max(-0.15,min(0.15,math.atan2(estimate.left_m,estimate.forward_m)))
    speed=min(0.06,(estimate.forward_m-stand_off_m)*0.2)
    return SkillCommand('approach',speed,yaw,'visible_target')
