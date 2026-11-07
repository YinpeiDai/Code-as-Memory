## RoboMME-Plus protocol difference

Use the same sensor, H5 and action interface as RoboMME. The official Plus
builder resolves native episode limits and task variants from its pinned metadata;
CaM does not replace these limits with RoboMME's 1300-step argument. Training IDs
are mapped to the selected metadata records before environment creation. The
run manifest declares whether perceptual variations are enabled. This adapter
remains separate from the shared learning prompt, policy memory and accounting.
