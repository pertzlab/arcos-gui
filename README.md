# arcos-gui

[![License](https://img.shields.io/pypi/l/arcos-gui.svg?color=green)](https://github.com/bgraedel/arcos-gui/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/arcos-gui.svg?color=green)](https://pypi.org/project/arcos-gui)
[![Python Version](https://img.shields.io/pypi/pyversions/arcos-gui.svg?color=green)](https://python.org)
[![tests](https://github.com/bgraedel/arcos-gui/workflows/tests/badge.svg)](https://github.com/bgraedel/arcos-gui/actions)
[![codecov](https://codecov.io/gh/bgraedel/arcos-gui/branch/main/graph/badge.svg)](https://codecov.io/gh/bgraedel/arcos-gui)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/arcos-gui)](https://napari-hub.org/plugins/arcos-gui)

A napari plugin to detect and visualize collective signaling events

----------------------------------
- Package specific Documentation: <https://bgraedel.github.io/arcos-gui>
- ARCOS documentation: <https://arcos.gitbook.io>

**A**utomated **R**ecognition of **C**ollective **S**ignalling (ARCOS) is an algorithm to identify collective spatial events in time series data,
that was written by Maciej Dobrzynski (https://github.com/dmattek). It is available as an R (ARCOS) and python (arcos4py) package.
ARCOS can identify and visualize collective protein activation in 2- and 3D cell cultures over time.

This plugin integrates ARCOS into napari. Users can import tracked time-series data in CSV format. The plugin
provides GUI elements to process this data with ARCOS. Layers containing the detected collective events are subsequently added to the viewer.

Following analysis, the user can export the output as a CSV file with the detected collective events or as a sequence of images to generate a movie.


![](https://user-images.githubusercontent.com/100028238/177808096-399abe6c-37a7-473b-96d2-afda5042a51e.gif)

[Watch full demo on youtube](https://www.youtube.com/watch?v=hG_z_BFcAiQ)


# Installation

You can install `arcos-gui` via [pip]:

    pip install arcos-gui

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"arcos-gui" is free and open-source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/bgraedel/arcos-gui/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/arcos-gui/
[PyPI]: https://pypi.org/

# Credits

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.
