"""Create a USD calibration target with standard OpenUSD material APIs."""
from pathlib import Path


def create_gaussian_scene(folder):
    """Author an explicit, uncalibrated visual placement using standard USD."""
    import json
    import numpy as np
    from pxr import Usd, UsdGeom, Gf
    folder=Path(folder)
    info=json.loads((folder/'report.json').read_text())
    low,high=np.array(info['bounds']['min']),np.array(info['bounds']['max'])
    if not np.isfinite([low,high]).all() or np.max(high-low)<=0:raise ValueError('Invalid Gaussian bounds')
    scale=.7/float(np.max(high-low));center=(high+low)/2
    stage=Usd.Stage.CreateNew(str(folder/'placed.usda'))
    UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    root=UsdGeom.Xform.Define(stage,'/Capture');stage.SetDefaultPrim(root.GetPrim())
    # AssetBase owns the spawn-root transform. Keep authored placement below it.
    placement=UsdGeom.Xform.Define(stage,'/Capture/Placement')
    placement.AddTranslateOp().Set(Gf.Vec3d(1.6,.45,.4))
    placement.AddScaleOp().Set(Gf.Vec3f(scale))
    placement.AddTranslateOp(opSuffix='center').Set(Gf.Vec3d(*(-center)))
    child=stage.DefinePrim('/Capture/Placement/Asset','Xform');child.GetReferences().AddReference('./scene.usdz')
    stage.GetRootLayer().Save()
    (folder/'placement.json').write_text(json.dumps(dict(scale=scale,source_center=center.tolist(),target_center_m=[1.6,.45,.4],scope='Arbitrary 0.7 m visual training prop; no measured reconstruction scale, collision or room claim.'),indent=2))
    return folder/'placed.usda'


def inspect_gaussian_placement(stage, prefix='/World/envs/env_0/Capture'):
    """Check the composed native-stage transform, not just the source USD."""
    import itertools
    import numpy as np
    from pxr import UsdGeom, Gf
    fields=[p for p in stage.Traverse() if str(p.GetPath()).startswith(prefix+'/') and p.GetTypeName()=='ParticleField3DGaussianSplat']
    if len(fields)!=1:raise ValueError('Expected one native Gaussian field')
    field=fields[0]
    points=np.asarray(field.GetAttribute('positions').Get())
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Invalid Gaussian positions')
    low,high=points.min(0),points.max(0)
    transform=UsdGeom.XformCache().GetLocalToWorldTransform(field)
    corners=np.array([transform.Transform(Gf.Vec3d(*map(float,p))) for p in itertools.product(*zip(low,high))])
    lo,hi=corners.min(0),corners.max(0);center=(lo+hi)/2
    error=float(np.max(np.abs(center-np.array([1.6,.45,.4]))))
    span=float(np.max(hi-lo))
    return dict(status='pass' if error<1e-4 and abs(span-.7)<1e-4 else 'fail',
                field=str(field.GetPath()),gaussian_count=len(points),world_bounds_m=[lo.tolist(),hi.tolist()],
                world_center_m=center.tolist(),center_error_m=error,max_span_m=span,
                scope='Composed stage placement only; camera pixels still require inspection; no collision or metric registration claim.')


def create_target(output):
    import cv2
    import numpy as np
    from pxr import Usd, UsdGeom, UsdShade, Sdf
    output=Path(output);output.mkdir(exist_ok=False)
    texture=np.full((240,240),255,dtype=np.uint8)
    texture[20:220,20:220]=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),7,200)
    cv2.imwrite(str(output/'marker.png'),texture)
    stage=Usd.Stage.CreateNew(str(output/'target.usda'))
    UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    root=UsdGeom.Xform.Define(stage,'/Target');stage.SetDefaultPrim(root.GetPrim())
    mesh=UsdGeom.Mesh.Define(stage,'/Target/Marker')
    mesh.CreatePointsAttr([(1,.12,.45),(1,-.12,.45),(1,-.12,.21),(1,.12,.21)])
    mesh.CreateFaceVertexCountsAttr([4]);mesh.CreateFaceVertexIndicesAttr([0,1,2,3]);mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr('none')
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set([(0,1),(1,1),(1,0),(0,0)])
    material=UsdShade.Material.Define(stage,'/Target/Material')
    shader=UsdShade.Shader.Define(stage,'/Target/Material/Preview');shader.CreateIdAttr('UsdPreviewSurface')
    uv=UsdShade.Shader.Define(stage,'/Target/Material/UV');uv.CreateIdAttr('UsdPrimvarReader_float2');uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');uv.CreateOutput('result',Sdf.ValueTypeNames.Float2)
    tex=UsdShade.Shader.Define(stage,'/Target/Material/Texture');tex.CreateIdAttr('UsdUVTexture');tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set('./marker.png')
    tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result');tex.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateOutput('surface',Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    return output/'target.usda'
