# Git workflow

This project uses the [GitHub Flow](https://guides.github.com/introduction/flow/) branching workflow. Hence, the rolling
release branch is `master`.

## Branches

We recommend the following naming conventions for branches:

- `feature/*` for developing new features and refactoring.
- `fix/*` for bug fixes.
- `release/*` for preparing releases.

## Testing

DAX is shipped with unit tests and default configurations for mypy type checking and flake8 style checking. Unit tests
are based on the standard Python unit test library. To run unit tests, use pytest. pytest, mypy, and flake8 need to be
installed separately. Tests can be executed by using the following commands:

```shell
$ pytest
$ mypy
$ flake8
```

## Merging

Sign off your patches using `git commit --signoff`. To merge into master, please issue a merge request.

## Releases

The `master` branch will always contain the latest production code (rolling release). Releases are tags and have a two
or three-component version number. See the following examples.

- `v6.0` for a major release targeting a new version of ARTIQ.
- `v6.1` for a feature release.
- `v6.1.1` for a minor release containing bug fixes.

Starting from `v6.0`, the major version number of DAX will match the version of the targeted ARTIQ release.
