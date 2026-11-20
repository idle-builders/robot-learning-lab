# Phone capture for the owner room

Capture procedure researched 26 October 2026. Scope: the room, furniture and
kitchen doorway along the intended robot route, predominantly at about 50 cm
camera height. A whole-flat walkthrough is not needed.

The original MOV metadata identifies an Apple iPhone 16 (not Pro). Assuming the
same phone, use RGB plus synchronized motion sensors; this model has no rear
LiDAR scanner. Do not ask the owner to buy another phone before inspecting a
small sensor recording. Both existing reconstructions remain unsatisfactory;
neither is a recommended replacement world. The longer video's selected map
connects 257/469 training images. This establishes missing camera coverage,
not that every original image is unusable or that capture caused every artifact.

## First deliverable: one small sensor recording

Install [Spectacular Rec](https://apps.apple.com/app/spectacular-rec/id6473188128)
by Spectacular AI. It records video and IMU; later computer processing estimates
the camera trajectory. It works on iPhones without LiDAR. This is the recommended
next capture method, not a claim that its export has already been integrated or
that it guarantees a good reconstruction.

Record one continuous 60–90 second loop covering a chair and the nearby kitchen
doorway. These durations, speeds and distances are practical starting choices
for this room, not vendor guarantees or automatic quality thresholds.

1. Clean the lens. Use bright, even light. Keep all lamps and curtains unchanged
   during capture; diffuse strong window light. Keep doors, chairs and objects
   stationary and people/pets outside the recorded view. Avoid a TV playing.
2. Use the app's normal sensor-recording mode and calibrated camera defaults.
   If camera selection is offered, start with the rear main camera at 1×. Keep
   the same camera and zoom throughout. Leave the phone in one orientation.
   Use 30 fps if configurable; do not force unsupported resolution or camera
   modes. Do not use cinematic blur, slow motion, or Action mode. If focus can
   be locked, first focus on furniture about 1–2 m away, check sharpness and
   keep it fixed for this room-scale test. Leave unexposed settings alone.
3. Hold the actual camera lens roughly 50 cm above the floor. Face a scene
   containing chair legs, furniture edges and a wall/floor junction. Keep a
   useful mix of nearby and background features in view; avoid filming only
   a blank wall, floor or ceiling. Exact surveyed height is not required for
   appearance, but report the measured approximate lens height.
4. Begin with 2–3 seconds nearly still. Move slowly in a shallow arc around
   the chair, roughly a metre over 5–10 seconds. Keep the chair visible while
   its legs change relative to the background. Move the camera sideways and
   forward; do not simply stand still and swivel it like a panorama.
5. Continue toward the doorway without cutting. Keep furniture or the jamb
   visible as you approach. Observe both sides of the opening from several
   translated positions. If the intended route goes through it, keep recording
   across the threshold and make a small loop in the visible kitchen area;
   do not abruptly swing from one room to another.
6. Return along a slightly offset path, revisiting the same chair and doorway.
   Finish near the starting position, facing the original scene. This gives
   overlapping visual evidence for reconnecting the trajectory. A shared
   starting point alone would not align independent recordings.
7. Review a few paused frames on the phone: chair legs and doorway edges should
   look reasonably crisp, with no prolonged focus hunting or exposure pumping.
   If motion smears the edges, use more light and move more slowly. Shorter
   exposure matters more here than merely increasing resolution or duration.

Do not make repeated passes along exactly the same line. The useful extra
information comes from seeing the same surfaces at slightly different positions.
The initial capture stays near the robot's intended viewpoints; a separate
higher overview may help geometry later but cannot replace the low views.

## References in the room

Measure the actual kitchen doorway clear opening between the inner jamb faces;
the owner's nominal 80 cm is currently only a prior. Also mark two fixed points
exactly 1.00 m apart on the floor with small contrasting tape marks, and include
them from several positions. Record which marks define that distance. These are
scale checks; neither the marks nor IMU data automatically establish a correct
metric floor frame.

Natural furniture texture usually supplies visual features. For a very plain
area, tape two or three different printed/textured sheets to fixed surfaces so
they are visible alongside furniture; keep them unmoved for all related captures.
No special reference target or identical starting pose is required. Do not rely
on mirrors, shiny surfaces or a bright window as the main visual anchor.

## Export the complete recording

Use the app's recording export/share to transfer the complete recording folder
or ZIP to the Mac, preferably with AirDrop or Files. Keep all original files and
names together. A typical Spectacular recording includes video, `data.jsonl`,
`calibration.json` and possibly `vio_config.yaml` or other auxiliary files. The
actual export contents must be inspected; depth is device/mode dependent and is
not expected from the rear camera on the standard iPhone 16.

Do not export just a Photos movie, trim the movie away from its metadata, or
send a messaging-app-compressed copy. Retain the original export. Also provide
the measured doorway width, reference distance and approximate lens height.

Before requesting a full-room capture, inspect the sample's synchronized video,
IMU, calibration and timestamps, then estimate/review its camera path and sparse
scene. Look for continuous coverage, trajectory breaks, drift at the return,
sharp overlapping images, furniture visibility and agreement with measured
references. Explain weaknesses with actual images and trajectory evidence.
These are qualitative assessments with measurements, not one automatic score.
Do not launch another expensive splat fit merely because the archive imports.
Runpod requests and training remain stopped unless separately resumed.

## Expand only after reviewing the pilot

For this bounded room, aim for one connected capture of roughly 2–4 minutes,
covering the useful route and returning to the starting view. Duration is a
planning estimate, not a quality target. Extra clips may cover missing areas,
but each should share a substantial stretch of recognizable furniture and
doorway views with the main capture—roughly 10–20 seconds of common scene is a
starting suggestion, not a guarantee. Keep lighting and the room layout the same.
Each new recording has its own sensor-coordinate frame until alignment is
established. Multiple clips that merely start at the same spot do not solve this.

## Primary sources

- [Spectacular AI mapping tools](https://www.spectacularai.com/mapping): supported
  phones, camera/IMU fusion, optional depth, computer-side processing.
- [Spectacular recording format](https://spectacularai.github.io/docs/sdk/recording.html):
  video, synchronized sensors and calibration files.
- [Spectacular Rec App Store listing](https://apps.apple.com/app/spectacular-rec/id6473188128):
  recorder role and manual-focus support; exact UI options require checking on
  the owner's installed version.
- [COLMAP capture guidance](https://colmap.github.io/tutorial.html#structure-from-motion):
  texture, similar lighting, overlap and translated viewpoints rather than pure
  camera rotation.
- [Nerfstudio custom data](https://docs.nerf.studio/quickstart/custom_dataset.html):
  supported sensor-recording routes, camera-pose requirement and motion-blur advice.
- [Apple iPhone 16 specifications](https://www.apple.com/iphone-16/specs/): camera
  and sensor capabilities.
