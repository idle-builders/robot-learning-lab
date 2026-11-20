// Every source-file link and fetch in the dashboard goes through this one helper.
// The local server answers /asset; a read-only web snapshot (static.js) maps the
// same path to its exported web copy instead.
export const assetUrl = path => globalThis.snapshotAssetUrl?.(path)
  ?? `/asset?path=${encodeURIComponent(path)}${globalThis.assetVersionQuery?.(path) ?? ''}`;
