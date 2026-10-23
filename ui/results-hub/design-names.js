// Mechanism identity is separate from designer provenance and the version selector.
export function designStudyName(row) {
  let name = String(row.title || '').replace(/^[A-C]\s*·\s*/, '').trim();
  const version = String(row.study_version || row.version || '').trim();
  if (version) {
    const escaped = version.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    name = name.replace(new RegExp(`\\s*[·|]\\s*(?:simplification\\s+)?${escaped}(?=\\s*(?:[·|]|$))`, 'i'), '');
  }
  return name.trim();
}
