# Architecture Visual Narratives (Shroud Viewer)

This viewer now includes architecture-specific narrative presets that remap rails, guide labels, zone callouts, and particle colors to tell the dataflow story of each model type.

## Presets

- `JEPA`
- `Text Diffusion`
- `H-net Tokenization`
- `SSM / E2E-TTT / Long-Context`
- `Adapters on Random Linear Maps`
- `Midnight 12L`

## How To Use

1. Open `shroud_viewer.html` in your normal Shroud workflow.
2. In the HUD, use `Architecture Narrative` dropdown to switch presets.
3. Toggle `View` (`Flow`, `Loop`, `Morph`, `Morph Timeline`, `Overlay`) to inspect the same trace from different narrative projections.
4. Hover particles and packet flows to inspect metadata while preset narrative colors and lanes remain active.

## URL Parameter

You can preselect a preset with query param `arch`:

- `?arch=jepa`
- `?arch=text_diffusion`
- `?arch=hnet_tokenization`
- `?arch=ssm_ttt_long_context`
- `?arch=adapters_random_linear_maps`
- `?arch=midnight_12l`

Aliases supported:

- `ssm_e2e_ttt_long_context` -> `ssm_ttt_long_context`
- `adapters_on_random_linear_maps` -> `adapters_random_linear_maps`
- `h_net_tokenization` -> `hnet_tokenization`
- `diffusion` -> `text_diffusion`
- `midnight12l` -> `midnight_12l`
- `midnight_12` -> `midnight_12l`
- `midnight` -> `midnight_12l`

## Notes

- Presets are fully optional; `default` keeps native Shroud behavior.
- Presets are optimized for the ratrod/nightcrawler layout profile and remain responsive on desktop and mobile.
