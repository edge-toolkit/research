# split-learning-mit

This directory contains the generated `mise.toml` for the `split-learning-mit` scenario.

The scenario exposes these workflow modules: split-learning-server.

This scenario is rendered against released artifacts rather than builds of this repository:
the images it names are published, and the binaries it runs are the released ones. Only the
image carrying its own module set is still built locally, since no release can hold a module
set particular to one deployment.

`secrets.env` holds the scenario's derived OpenObserve and OTLP credentials. It is derived from the
scenario input, so regenerating this deployment rewrites it; if it is missing, regenerate before
starting the stack. A deployment generated outside the repository is written with a `.gitignore`
covering it, so its credential is not committed by whatever repository it lands in.

## Run With Mise

Fetch the binaries and module packages the tasks below name before the first run.
`GITHUB_TOKEN` has to be set: GitHub Packages rejects an unauthenticated read even for a public
package. The registry configuration is exported rather than relied on from `mise.toml`, because
mise does not apply its own `[env]` to the resolution this command performs:

```bash
NPM_CONFIG_USERCONFIG="$PWD/npmrc" mise install
```

From this directory, start the scenario with:

```bash
mise run generated-scenario
```

That task starts both OpenObserve and `ws-server` for this scenario.

### Open The OpenObserve UI

From this directory, open the OpenObserve UI with:

```bash
mise run open-o2
```
