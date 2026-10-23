import * as THREE from 'three';

// Native manifests express joint pivots in the neutral world frame, and parts
// retain neutral world coordinates. Parent transforms compose from root to tip.
export function createJointRig(model, metadata, {scale = .001, limits = {}} = {}) {
  const joints = (metadata.joints || []).filter(j =>
    Array.isArray(j.origin_world_mm || j.origin_mm) && Array.isArray(j.axis_world_neutral || j.axis));
  // A loop closure cannot be replaced by independently rotating its links.
  // Such sources retain their exact native saved-configuration playback.
  if (new Set(joints.map(j => j.child)).size !== joints.length) return null;
  const driven = j => j.driven_by || (j.coupled_to ? {linear: {[j.coupled_to]: j.multiplier}} : null);
  const byChild = new Map(joints.map(j => [j.child, j]));
  const clean = name => THREE.PropertyBinding.sanitizeNodeName(name);
  const partByName = new Map((metadata.parts || []).map(p => [clean(p.name), p]));
  model.updateMatrixWorld(true);
  const inverse = model.matrixWorld.clone().invert(), nodes = [];
  model.traverse(node => {
    const part = partByName.get(node.name);
    if (part?.owning_link && !partByName.has(node.parent?.name))
      nodes.push({node, link: part.owning_link, base: inverse.clone().multiply(node.matrixWorld)});
  });
  const movableLinks = new Set(nodes.map(n => n.link));
  const ancestry = link => { const seen = new Set(); while (byChild.has(link) && !seen.has(link)) {seen.add(link); link = byChild.get(link).parent;} return seen; };
  const scopedRange = (name, native) => {
    const saved=limits[name];
    if(!Array.isArray(saved))return native;
    if(!Array.isArray(native))return saved;
    return [Math.max(native[0],saved[0]),Math.min(native[1],saved[1])];
  };
  const validRange = range => Array.isArray(range) && range.length===2 && range.every(Number.isFinite) && range[0]<range[1];
  const nativeControls = joints.filter(j => !driven(j) && j.type !== 'prismatic' &&
    validRange(scopedRange(j.name,j.range_deg)) && [...movableLinks].some(link => ancestry(link).has(j.child)))
    .map(j => ({name: j.name, range: scopedRange(j.name,j.range_deg)}));
  // A reduced closed mechanism can have a command coordinate without a physical
  // joint on that axis. Only expose an explicit, source-defined angular range.
  const controls = metadata.controls?.length ? metadata.controls.filter(c =>
    typeof c.name === 'string' && validRange(scopedRange(c.name,c.range_deg))
  ).map(c => ({name:c.name,range:scopedRange(c.name,c.range_deg)})) : nativeControls;
  const supported = joints.every(j => {const d = driven(j); return !d || d.linear || (d.dof && Array.isArray(d.table_deg) && d.table_deg.every(row => Array.isArray(row) && row.length >= 2));});
  if (!nodes.length || !controls.length || !supported) return null;
  function dependent(joint, values) {
    const spec = driven(joint);
    if (!spec) return values[joint.name] || 0;
    if (spec.linear) return Object.entries(spec.linear).reduce((sum, [key, gain]) => sum + (values[key] || 0) * gain, 0) + (spec.offset_deg || 0);
    const table = spec.table_deg, q = values[spec.dof] || 0;
    if (!Array.isArray(table) || !table.every(row => Array.isArray(row) && row.length >= 2)) return 0;
    if (q <= table[0][0]) return table[0][1];
    for (let i = 1; i < table.length; i++) if (q <= table[i][0]) { const [a,b] = table[i-1], [c,d] = table[i]; return b + (d-b)*(q-a)/(c-a); }
    return table.at(-1)[1];
  }
  function pose(values, {anchorLink = null, anchorLinksByRoot = null} = {}) {
    const matrices = new Map(), visiting = new Set();
    function transform(link) {
      if (matrices.has(link)) return matrices.get(link);
      if (visiting.has(link)) throw new Error('Cyclic native joint hierarchy');
      const joint = byChild.get(link);
      if (!joint) return new THREE.Matrix4();
      visiting.add(link);
      const origin = new THREE.Vector3(...(joint.origin_world_mm || joint.origin_mm)).multiplyScalar(scale);
      const axis = new THREE.Vector3(...(joint.axis_world_neutral || joint.axis)).normalize();
      const angle = dependent(joint, values);
      const local = joint.type === 'prismatic' ? new THREE.Matrix4().makeTranslation(...axis.multiplyScalar(angle * scale).toArray()) :
        new THREE.Matrix4().makeTranslation(...origin.toArray()).multiply(new THREE.Matrix4().makeRotationAxis(axis, THREE.MathUtils.degToRad(angle))).multiply(new THREE.Matrix4().makeTranslation(...origin.clone().negate().toArray()));
      const matrix = transform(joint.parent).clone().multiply(local);
      matrices.set(link, matrix); visiting.delete(link); return matrix;
    }
    const anchors = new Map();
    function anchorFor(link) {
      let root = link;
      const seen = new Set();
      while (byChild.has(root)) {
        if (seen.has(root)) throw new Error('Cyclic native joint hierarchy');
        seen.add(root); root = byChild.get(root).parent;
      }
      const name = anchorLinksByRoot?.[root] || anchorLink;
      if (!name) return new THREE.Matrix4();
      if (!anchors.has(name)) anchors.set(name, transform(name).clone().invert());
      return anchors.get(name);
    }
    for (const {node, link, base} of nodes) {
      const matrix = node.parent.matrixWorld.clone().invert().multiply(model.matrixWorld).multiply(anchorFor(link)).multiply(transform(link)).multiply(base);
      matrix.decompose(node.position, node.quaternion, node.scale); node.updateMatrix();
    }
    model.updateMatrixWorld(true);
  }
  const mappedReferenceParts = nodes.filter(({node}) => {
    const part = partByName.get(node.name);
    return part?.inspection_reference || part?.reference_only || part?.context;
  }).length;
  return {controls, pose, mappedParts: nodes.length, mappedReferenceParts, totalParts: (metadata.parts || []).length};
}
