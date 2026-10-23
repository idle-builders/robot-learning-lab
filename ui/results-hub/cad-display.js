// Body panels are distinct from purchased actuator cases and neighboring joints.
// New manifests should declare display_group; legacy panel roles stay usable.
export function displayGroup(part) {
  if (part?.display_group) return part.display_group;
  if (['motor','actuator','actuator_housing','motor_housing','gearbox'].includes(part?.role)) return 'mechanism';
  return ['shell','skin','body_shell','body_panel','body_guard'].includes(part?.role) ? 'ape_body_shell' : 'mechanism';
}
export function isReference(part) {
  return !!part?.inspection_reference || displayGroups(part).some(group => ['neighbor_reference','body_reference'].includes(group));
}
export function displayGroups(part) {
  return [...new Set([displayGroup(part), ...(Array.isArray(part?.display_groups) ? part.display_groups : [])])];
}
export function visibleDisplayGroups(part, {shells = true, references = false} = {}) {
  const groups = displayGroups(part);
  // Older body-panel references also carry the neighboring-joint group.
  // Their visibility belongs to the shell control, independently of neighbors.
  if (groups.includes('ape_body_shell')) return shells;
  return !groups.includes('neighbor_reference') || references;
}
