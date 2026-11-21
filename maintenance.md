
# PyPI upload

Package a Python Package/ version bump See: https://packaging.python.org/tutorials/packaging-projects/

Run the commands below from the repository root with a clean working tree.

## Install dependencies:
```bash
python -m pip install --upgrade build twine bump-my-version
```

[`bump-my-version`](https://github.com/callowayproject/bump-my-version) is the
maintained successor to `bump2version`. The command below updates the version,
creates a commit, and tags it.

```bash
rm -rf build/ dist/ pb_bss.egg-info/
git pull
bump-my-version bump --verbose --commit --tag patch  # major, minor or patch
python -m build --sdist
python -m twine check dist/*.tar.gz
python -m twine upload --repository testpypi dist/*.tar.gz
git push origin --tags
git push
python -m twine upload dist/*.tar.gz
```
