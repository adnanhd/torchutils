import math
import os
import datetime
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def makedirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


class Generator(object):
    parent_dir = './'

    def __init__(self, parent=None, folder=None):
        if parent is None:
            parent = self.parent_dir

        self.parent = parent
        self.folder = folder
        makedirs(self.path)

    @property
    def path(self):
        return os.path.normpath(os.path.join(self.parent, self.folder))


class FileGenerator(Generator):
    ending = None
    def __init__(self, buf, parent=None, folder=None, hierarchy=False, preserve=False):
        assert isinstance(buf, str)
        super(FileGenerator, self).__init__(parent=parent, folder=folder)
        self.fname = buf

        self._buffer = open(os.path.join(self.path, self.fname), 'w')
        self.hierarchy = hierarchy
        self.add_header()

    def __del__(self):
        self._buffer.close()

    def add_header(self):
        return NotImplementedError()

    def add_parent_page(self):
        return NotImplementedError()

    def add_dataframe(self):
        return NotImplementedError()

    def add_figure(self):
        return NotImplementedError()

    def add_module(self):
        return NotImplementedError()

    def __call__(self, *args):
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                self.add_dataframe(arg)
            elif isinstance(arg, FigureGenerator):
                self.add_figure(arg)
            elif isinstance(arg, torch.nn.Module):
                self.add_module(arg)


class WandbGenerator(Generator):
    def __init__(self, project=None, entity=None, model=None, **config):
        self.wandb = wandb.init(project=project, entity=entity, config=config)

        if model is not None:
            self.wandb.watch(self.model, log="all")

    def __call__(self, train_df=None, valid_df=None, test_df=None):
        pass


class LatexGenerator(FileGenerator):
    main_page = 'main.tex'
    ending = ".tex"

    def __init__(self, entity, project=None, model=None, **config):
        if not entity.endswith(self.ending):
            entity = entity + self.ending

        self.fname = entity.split(self.ending)[0]

        super(LatexGenerator, self).__init__(
            entity, project=project, folder='latex')

        self._buffer.write(r'\documentclass{article}')
        self._buffer.write(r'\usepackage[utf8]{inputenc}')
        self._buffer.write(r'\usepackage{booktabs}')

        self._buffer.write(r'\title{' + str(project) + '}')
        self._buffer.write(r'\author{' + str(entity) + '}')
        self._buffer.write(r'\date{December 2021}')

        self._buffer.write(r'\begin{document}')
        self._buffer.write(r'\maketitle')

    def __del__(self):
        self._buffer.write(r'\end{document}')
        super(LatexGenerator, self).__del__()

    def add_parent_page(self):
        with open(os.path.join(self.folder, self.main_page), 'w') as f:
            for each in self._instances:
                f.write(f'\\include{each.fname}')

    def add_dataframe(self, *dfs):
        for df in dfs:
            df.to_html(self._buffer)

    def add_image(self, path):
        self._buffer.write(r"\begin{figure}\includegraphics[width=\linewidth]{" + path
                           + r"}\caption{A boat.}\label{fig:boat1}\end{figure}")


class HTMLGenerator(FileGenerator):
    main_page = 'index.html'
    ending = ".html"
    from dominate import document, tags

    def __init__(self, fname, project=None, model=None, 
                    main_page=False, overwrite=True, **config):
        if not fname.endswith(self.ending):
            fname = fname + self.ending

        self.doc = self.document(title=fname.split(self.ending)[0])
        super(HTMLGenerator, self).__init__(fname, parent=project, folder='html', 
                                            hierarchy=main_page, preserve=not overwrite)
        
        self.basecontent = config

    def __del__(self):
        self._buffer.write(str(self.render()))

    def render(self, *args, **kwargs):
        return self.doc.render(*args, **kwargs)
    
    def add_title(self):
        pass

    def add_header(self, meta = {"content": "utf-8", "http-equiv": "encoding"}):
        self.doc.head.add(self.tags.meta(**meta))
        if self.hierarchy:
            self.doc.body.add(self.tags.a("back", href=self.main_page))
            self.add_parent_page()

    def add_parent_page(self):
        with open(os.path.join(self.path, self.main_page), 'w') as f:
            html_files = list(filter(lambda x: x.endswith(
                self.ending), os.listdir(self.path)))
            if self.main_page in html_files:
                html_files.remove(self.main_page)
            for each in html_files:
                fname = each.split(self.ending)[0]
                f.write(str(self.tags.a(fname, href=fname +
                                        self.ending)) + str(self.tags.br()))

    def add_dataframe(self, df: pd.DataFrame):
        self.doc.body.add_raw_string(df.to_html())

    def add_figure(self, fig, fname=None):
        if fname is None:
            self_fname = self.fname.split(self.ending)[0]
            fname = self_fname + fig.ending
        path = os.path.join(self.path, fname)
        fig.plot(path=path)
        self.doc.body.add(self.tags.img(src=fname))

    def add_module(self, model: torch.nn.Module):
        self.doc.body.add(self.tags.pre(str(model)))


class _AxisGenerator(Generator):
    def __init__(self, title=None, ylabel=None, xlabel=None):
        self.title = title
        self.ylabel = ylabel
        self.xlabel = xlabel

    def __call__(self, ax=None):
        if ax is None:
            ax = plt.axis()
        if self.title is not None:
            ax.set_title(self.title)
        if self.xlabel is not None:
            ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel)
        return ax


class FigureGenerator(Generator):
    ending = '.png'

    def __init__(self, *axes, fig=None, project=None, **kwargs):
        super(FigureGenerator, self).__init__(parent=project, folder='plot')

        if fig is None:
            self.fig = plt.figure()

        self.axes = self.fig.subplots(*axes, **kwargs)

        if isinstance(self.axes, plt.Axes):
            self.axes = np.array([self.axes])

    def __getitem__(self, index):
        return self.axes[index]

    def __len__(self):
        return len(self.axes)

    def __setitem__(self, index, axis):
        self.axes[index] = axis(self.axes[index])

    def plot(self, fname=None, path=None, *argv, **kwargs):
        if fname is None and path is None:
            plt.show()
        else:
            if path is None:
                path = self.path
            if fname is not None:
                path = os.path.normpath(os.path.join(path, fname))
            plt.savefig(path)

    def __del__(self):
        plt.close(self.fig)

    class Semilogy(_AxisGenerator):
        def __init__(self, df, title=None, xlabel=None, ylabel=None):
            self.df = df

            if title is None:
                title = f'Training and Validation Metrics in {len(df)} Iter'
            if xlabel is None:
                xlabel = 'Epochs'
            if ylabel is None:
                ylabel = 'Loss'

            super(self.__class__, self).__init__(
                title=title, ylabel=ylabel, xlabel=xlabel)

        def __call__(self, ax):
            # TODO: add 'x' and 'o' for train and valid
            super(self.__class__, self).__call__(ax=ax)
            ax.semilogy(self.df, 'o-')
            ax.legend(self.df.columns)
            return ax

    class Histogram(_AxisGenerator):
        def __init__(self, df, ax=None, title=None, xlabel=None, ylabel=None):
            self.df = df

            if title is None:
                title = f'Testing Metrics in {len(df)} Iter'
            if xlabel is None:
                xlabel = 'Metrics'
            if ylabel is None:
                ylabel = 'Cases'

            super(self.__class__, self).__init__(
                title=title, ylabel=ylabel, xlabel=xlabel)

        def __call__(self, ax):
            # TODO: add 'x' and 'o' for train and valid
            super(self.__class__, self).__call__(ax=ax)
            ax.hist(self.df, bins=int(math.log2(len(self.df) ** 2) + 1)
                    if len(self.df) else None)
            ax.legend(self.df.columns)
            return ax
