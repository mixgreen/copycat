import unittest
from unittest.mock import patch
import typing
import random
import argparse
import os.path
import h5py

from artiq.language.core import rpc, portable, kernel, host_only
import artiq.experiment
import artiq.master.worker_db
import artiq.tools

import dax.util.artiq
import dax.util.output


class ArtiqTestCase(unittest.TestCase):

    def test_get_managers(self):
        # Create an experiment object using the helper get_managers() function
        with dax.util.artiq.get_managers() as managers:
            self.assertIsInstance(artiq.experiment.EnvExperiment(managers), artiq.experiment.HasEnvironment)
            self.assertIsInstance(managers, tuple)
            self.assertIsInstance(managers, dax.util.artiq.CloseableManagersTuple)

            num_managers = 4
            self.assertEqual(len(managers), num_managers)

            # Test unpacking
            _, _, _, _ = managers
            # Test indexing
            for i in range(num_managers):
                _ = managers[i]

    def test_get_managers_dataset_db(self):
        with dax.util.output.temp_dir():
            dataset_db = 'dataset_db.pyon'
            key = 'foo'
            value = 99

            with open(dataset_db, mode='x') as f:
                # Write pyon file
                f.write(f'{{\n    "{key}": {value}\n}}')

            # Create environment
            with dax.util.artiq.get_managers(dataset_db=dataset_db) as managers:
                env = artiq.experiment.EnvExperiment(managers)
                self.assertEqual(env.get_dataset(key), value, 'Retrieved dataset did not match earlier set value')

    def _test_decorator_classifier(self, fn, name, *, undecorated=False, kernel_=False, rpc_=False, rpc_async=False,
                                   portable_=False, host_only_=False, **kwargs):
        function_list = [
            (self._undecorated_fn, undecorated, 'Undecorated'),
            (self._kernel_fn, kernel_, 'Kernel'),
            (self._rpc_fn, rpc_, 'RPC'),
            (self._rpc_async_fn, rpc_async, 'Async RPC'),
            (self._portable_fn, portable_, 'Portable'),
            (self._host_only_fn, host_only_, 'Host only'),
        ]
        for test_function, ref, test_name in function_list:
            self.assertEqual(fn(test_function, **kwargs), ref, f'{test_name} function wrongly classified by {name}')

        self.assertFalse(fn(None))  # This is allowed

    def test_is_kernel(self):
        self._test_decorator_classifier(dax.util.artiq.is_kernel, 'is_kernel()', kernel_=True)

    def test_is_portable(self):
        self._test_decorator_classifier(dax.util.artiq.is_portable, 'is_portable()', portable_=True)

    def test_is_host_only(self):
        self._test_decorator_classifier(dax.util.artiq.is_host_only, 'is_host_only()', host_only_=True)

    def test_is_rpc(self):
        self._test_decorator_classifier(dax.util.artiq.is_rpc, 'is_rpc()', rpc_=True, rpc_async=True)

    def test_is_async_rpc(self):
        self._test_decorator_classifier(dax.util.artiq.is_rpc, 'is_rpc() (async)', rpc_async=True, flags={'async'})

    def test_is_decorated(self):
        self._test_decorator_classifier(dax.util.artiq.is_decorated, 'is_decorated()',
                                        kernel_=True, rpc_=True, rpc_async=True, portable_=True, host_only_=True)

    def test_default_enumeration_value(self):
        data = [
            ([], artiq.experiment.NoDefault, None),
            (['foo'], artiq.experiment.NoDefault, 'foo'),
            (['foo'], 'foo', 'foo'),
            (['foo', 'bar'], artiq.experiment.NoDefault, None),
            (['foo', 'bar'], 'bar', 'bar'),
        ]

        for choices, default, returned_default in data:
            e = dax.util.artiq.default_enumeration_value(choices, default)
            self.assertListEqual(e.choices, choices)
            if returned_default is None:
                with self.assertRaises(artiq.experiment.DefaultMissing):
                    e.default()
            else:
                self.assertEqual(e.default(), returned_default)

    def test_process_arguments(self):
        arguments = {'foo': 1,
                     'range': artiq.experiment.RangeScan(1, 10, 9),
                     'center': artiq.experiment.CenterScan(1, 10, 9),
                     'explicit': artiq.experiment.ExplicitScan([1, 10, 9]),
                     'no': artiq.experiment.NoScan(10)}

        processed_arguments = dax.util.artiq.process_arguments(arguments)
        self.assertEqual(len(arguments), len(processed_arguments))
        self.assertIsNot(arguments, processed_arguments)
        for v in processed_arguments.values():
            self.assertNotIsInstance(v, artiq.experiment.ScanObject)
        self.assertDictEqual(processed_arguments, {k: v.describe() if isinstance(v, artiq.experiment.ScanObject) else v
                                                   for k, v in arguments.items()})

    def test_parse_arguments(self):
        arguments = {'foo': 1,
                     'range': artiq.experiment.RangeScan(1, 10, 9),
                     'center': artiq.experiment.CenterScan(1, 10, 9),
                     'explicit': artiq.experiment.ExplicitScan([1, 10, 9]),
                     'no': artiq.experiment.NoScan(10),
                     'bar': 'baz',
                     'foobar': 0.345,
                     'baz': [1, 2, 3]}

        processed_arguments = dax.util.artiq.process_arguments(arguments)
        commandline_arguments = [f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}'
                                 for k, v in processed_arguments.items()]

        parser = argparse.ArgumentParser()
        parser.add_argument('args', nargs='*')
        unprocessed_arguments = parser.parse_args(commandline_arguments).args
        parsed_arguments = artiq.tools.parse_arguments(unprocessed_arguments)

        self.assertDictEqual(parsed_arguments, processed_arguments)

    def test_managers_tuple(self):
        managers = dax.util.artiq.get_managers()
        with patch.object(managers.device_mgr, 'close_devices') as managers_fn:
            with managers:
                with dax.util.artiq.clone_managers(managers) as clone:
                    self.assertIsInstance(clone, dax.util.artiq.ManagersTuple)
                    with patch.object(clone.device_mgr, 'close_devices') as clone_fn:
                        with clone:
                            pass
                        self.assertFalse(clone_fn.called)
                self.assertFalse(managers_fn.called)
            self.assertTrue(managers_fn.called)

    def test_closable_managers_tuple(self):
        m = dax.util.artiq.get_managers()
        self.assertIsInstance(m, dax.util.artiq.CloseableManagersTuple)

        with patch.object(m.device_mgr, 'close_devices') as fn:
            m.close()
            self.assertTrue(fn.called, 'Devices were not closed')

    def test_closable_managers_tuple_context(self):
        m = dax.util.artiq.get_managers()
        self.assertIsInstance(m, dax.util.artiq.CloseableManagersTuple)

        with patch.object(m.device_mgr, 'close_devices') as fn:
            with m:
                pass
            self.assertTrue(fn.called, 'Devices were not closed')

    def test_managers_tuple_write_hdf5(self):
        m = dax.util.artiq.get_managers()
        default_meta_keys = ['artiq_version', 'dax_version']
        default_groups = ['archive', 'datasets']

        with m:
            for meta in [{}, {'foo': 1, 'bar': 2.0}]:
                with dax.util.output.temp_dir() as tmp:
                    file_name = os.path.join(tmp, 'out.h5')
                    self.assertFalse(os.path.isfile(file_name))
                    m.write_hdf5(file_name, metadata=meta)
                    self.assertTrue(os.path.isfile(file_name))

                    result = h5py.File(file_name, mode='r')
                    for k, v in meta.items():
                        self.assertEqual(result[k][()], v)
                    for k in default_meta_keys + default_groups:
                        self.assertIn(k, result)
                    self.assertEqual(len(result), len(meta) + len(default_meta_keys) + len(default_groups),
                                     'Found more keys than expected')

    def test_clone_managers(self):
        with dax.util.artiq.get_managers() as managers:
            with dax.util.artiq.clone_managers(managers) as cloned:
                self.assertIsInstance(cloned, dax.util.artiq.ManagersTuple,
                                      'Cloned managers tuple is not of the correct type (should not be closable)')
                self.assertIs(managers.device_mgr, cloned.device_mgr, 'Device manager was modified unintentionally')
                self.assertIsNot(managers.dataset_mgr, cloned.dataset_mgr, 'Dataset manager was not replaced')
                self.assertIsInstance(cloned.dataset_mgr, artiq.master.worker_db.DatasetManager)
                self.assertIsNot(managers.argument_mgr, cloned.argument_mgr, 'Argument manager was not replaced')
                self.assertFalse(cloned.argument_mgr.unprocessed_arguments, 'Arguments not decoupled')
                self.assertIsNot(managers.scheduler_defaults, cloned.scheduler_defaults,
                                 'Scheduler defaults were not replaced')

    def test_clone_managers_recursive(self):
        with dax.util.artiq.get_managers() as managers:
            dataset_db = managers.dataset_mgr.ddb
            clone = managers
            for _ in range(3):
                clone = dax.util.artiq.clone_managers(clone)
                self.assertIs(dataset_db, clone.dataset_mgr.ddb)

    def test_clone_managers_arguments(self):
        with dax.util.artiq.get_managers() as managers:
            arguments = {'foo-bar': 1, 'bar-baz': 4, 'name': 'some_name'}
            kwargs = {'foo': 4.4, 'bar': 'bar'}
            ref = arguments.copy()  # Copy a reference for usage later

            with dax.util.artiq.clone_managers(managers, arguments=arguments, **kwargs) as cloned:
                # Check if we did not accidentally mutated the original arguments dict
                self.assertDictEqual(ref, arguments, 'The original given arguments were mutated')

                # Update reference to match expected outcome
                ref.update(kwargs)
                self.assertDictEqual(ref, cloned.argument_mgr.unprocessed_arguments,
                                     'Arguments were not passed correctly')

    def test_clone_managers_dataset_db_broadcast(self):
        dataset_kwargs = [{'broadcast': b, 'persist': p} for b in [False, True] for p in [False, True]]
        self.assertEqual(len(dataset_kwargs), 4)
        rng = random.Random()

        with dax.util.artiq.get_managers() as managers:
            with dax.util.artiq.clone_managers(managers) as cloned:
                class TestExperiment(artiq.experiment.EnvExperiment):
                    def run(self):
                        pass

                # Create the main experiment
                exp = TestExperiment(managers)
                cloned_exp = TestExperiment(cloned)

                # Test write-read from main to cloned experiment
                for i, kwargs in enumerate(dataset_kwargs):
                    key = f'main_to_cloned{i}'
                    value = rng.random()
                    exp.set_dataset(key, value, **kwargs)
                    if any(kwargs.values()):
                        self.assertEqual(value, cloned_exp.get_dataset(key))
                    else:
                        with self.assertRaises(KeyError, msg='Unexpected leakage of datasets'):
                            cloned_exp.get_dataset(key)

                # Test write-read from cloned to main experiment
                for i, kwargs in enumerate(dataset_kwargs):
                    key = f'cloned_to_main{i}'
                    value = rng.random()
                    cloned_exp.set_dataset(key, value, **kwargs)
                    if any(kwargs.values()):
                        self.assertEqual(value, exp.get_dataset(key))
                    else:
                        with self.assertRaises(KeyError, msg='Unexpected leakage of datasets'):
                            exp.get_dataset(key)

    def test_isolate_managers(self):
        with dax.util.artiq.get_managers(arguments={'foo': 1, 'bar': 2}) as managers:
            with dax.util.artiq.isolate_managers(managers) as isolated:
                self.assertIsInstance(isolated, dax.util.artiq.ManagersTuple,
                                      'Isolated managers tuple is not of the correct type (should not be closable)')
                self.assertEqual(len(managers), len(isolated))
                for m, i in zip(managers, isolated):
                    if not isinstance(m, dict):
                        self.assertNotEqual(m, i)
                    else:
                        self.assertFalse(i)
                    self.assertIsNot(m, i)
                self.assertFalse(isolated.argument_mgr.unprocessed_arguments, 'Arguments not decoupled')

    def test_isolate_managers_recursive(self):
        with dax.util.artiq.get_managers() as managers:
            dataset_dbs = [managers.dataset_mgr.ddb]
            isolated = managers
            for _ in range(3):
                isolated = dax.util.artiq.isolate_managers(isolated)
                for ddb in dataset_dbs:
                    self.assertIsNot(ddb, isolated.dataset_mgr.ddb)
                dataset_dbs.append(isolated.dataset_mgr.ddb)

    def test_isolate_managers_arguments(self):
        with dax.util.artiq.get_managers() as managers:
            arguments = {'foo-bar': 1, 'bar-baz': 4, 'name': 'some_name'}
            kwargs = {'foo': 4.4, 'bar': 'bar'}
            ref = arguments.copy()  # Copy a reference for usage later

            with dax.util.artiq.clone_managers(managers, arguments=arguments, **kwargs) as isolated:
                # Check if we did not accidentally mutated the original arguments dict
                self.assertDictEqual(ref, arguments, 'The original given arguments were mutated')

                # Update reference to match expected outcome
                ref.update(kwargs)
                self.assertDictEqual(ref, isolated.argument_mgr.unprocessed_arguments,
                                     'Arguments were not passed correctly')

    def test_isolate_managers_device_db(self):
        ddb: typing.Dict[str, typing.Any] = {
            'core': {
                'type': 'local',
                'module': 'artiq.coredevice.core',
                'class': 'Core',
                'arguments': {'host': None, 'ref_period': 1e-9}
            },
            'core_cache': {
                'type': 'local',
                'module': 'artiq.coredevice.cache',
                'class': 'CoreCache'
            },
            'core_dma': {
                'type': 'local',
                'module': 'artiq.coredevice.dma',
                'class': 'CoreDMA'
            },
        }

        with dax.util.artiq.get_managers(device_db=ddb) as managers:
            with dax.util.artiq.isolate_managers(managers) as isolated:
                class TestExperiment(artiq.experiment.EnvExperiment):
                    # noinspection PyMethodParameters
                    def build(self_):
                        # Test if device keys raise, must be in the build function
                        for key in ddb:
                            with self.assertRaises(artiq.master.worker_db.DeviceError,
                                                   msg=f'Device key "{key}" did not raise'):
                                self_.get_device(key)

                    def run(self):
                        pass

                # Create the test experiment
                exp = TestExperiment(isolated)

                # Test if device DB is empty
                self.assertFalse(exp.get_device_db())

    def test_isolate_managers_dataset_db(self):
        with dax.util.artiq.get_managers() as managers:
            with dax.util.artiq.isolate_managers(managers) as isolated:
                self.assertIsNot(managers.dataset_mgr.ddb, isolated.dataset_mgr.ddb)
                self.assertFalse(isolated.dataset_mgr.ddb)

    def test_isolate_managers_dataset_db_broadcast(self):
        dataset_kwargs = [{'broadcast': b, 'persist': p} for b in [False, True] for p in [False, True]]
        self.assertEqual(len(dataset_kwargs), 4)

        with dax.util.artiq.get_managers() as managers:
            with dax.util.artiq.isolate_managers(managers) as isolated:
                class TestExperiment(artiq.experiment.EnvExperiment):
                    def run(self):
                        pass

                # Create the main experiment
                exp = TestExperiment(managers)
                isolated_exp = TestExperiment(isolated)

                # Test write-read from main to isolated experiment
                for i, kwargs in enumerate(dataset_kwargs):
                    key = f'main_to_isolated{i}'
                    exp.set_dataset(key, 33, **kwargs)
                    with self.assertRaises(KeyError, msg='Datasets not isolated'):
                        isolated_exp.get_dataset(key)

                # Test write-read from isolated to main experiment
                for i, kwargs in enumerate(dataset_kwargs):
                    key = f'isolated_to_main{i}'
                    isolated_exp.set_dataset(key, 33, **kwargs)
                    with self.assertRaises(KeyError, msg='Datasets not isolated'):
                        exp.get_dataset(key)

    def test_pause_strict_priority(self):
        # Must be host only
        self.assertTrue(dax.util.artiq.is_host_only(dax.util.artiq.pause_strict_priority))

        with dax.util.artiq.get_managers() as managers:
            class TestExperiment(artiq.experiment.EnvExperiment):
                def build(self):
                    self.core = self.get_device('core')
                    self.scheduler = self.get_device('scheduler')

                # noinspection PyMethodParameters
                def run(self_):
                    dax.util.artiq.pause_strict_priority(self_.scheduler)
                    dax.util.artiq.pause_strict_priority(self_.scheduler, polling_period=0)
                    with self.assertRaises(TypeError, msg='Wrong scheduler object did not raise'):
                        dax.util.artiq.pause_strict_priority(self_.core)
                    with self.assertRaises(ValueError, msg='Negative polling period did not raise'):
                        dax.util.artiq.terminate_running_instances(self_.scheduler, polling_period=-1)

            # Create the main experiment
            exp = TestExperiment(managers)
            exp.run()

    def test_terminate_running_instances(self):
        # Must be host only
        self.assertTrue(dax.util.artiq.is_host_only(dax.util.artiq.terminate_running_instances))

        with dax.util.artiq.get_managers() as managers:
            class TestExperiment(artiq.experiment.EnvExperiment):
                def build(self):
                    self.core = self.get_device('core')
                    self.scheduler = self.get_device('scheduler')

                # noinspection PyMethodParameters
                def run(self_):
                    dax.util.artiq.terminate_running_instances(self_.scheduler)
                    dax.util.artiq.terminate_running_instances(self_.scheduler, timeout=0, polling_period=0)
                    with self.assertRaises(TypeError, msg='Wrong scheduler object did not raise'):
                        dax.util.artiq.terminate_running_instances(self_.core)
                    with self.assertRaises(ValueError, msg='Negative timeout did not raise'):
                        dax.util.artiq.terminate_running_instances(self_.scheduler, timeout=-1)
                    with self.assertRaises(ValueError, msg='Negative polling period did not raise'):
                        dax.util.artiq.terminate_running_instances(self_.scheduler, polling_period=-1)

            # Create the main experiment
            exp = TestExperiment(managers)
            exp.run()

    """Functions used for tests"""

    def _undecorated_fn(self):
        pass

    @rpc
    def _rpc_fn(self):
        pass

    @rpc(flags={'async'})
    def _rpc_async_fn(self):
        pass

    @portable
    def _portable_fn(self):
        pass

    @kernel
    def _kernel_fn(self):
        pass

    @host_only
    def _host_only_fn(self):
        pass


class DelayBuildTestCase(unittest.TestCase):
    def test_delay_build(self):
        build_called = False
        build_args = None
        build_kwargs = None
        prepare_called = False
        run_called = False
        analyze_called = False

        @dax.util.artiq.delay_build
        class _TestExperiment(artiq.experiment.EnvExperiment):
            def build(self, *args, **kwargs):
                nonlocal build_called, build_args, build_kwargs
                build_called = True
                build_args = args
                build_kwargs = kwargs

            def prepare(self):
                nonlocal prepare_called
                prepare_called = True

            def run(self):
                nonlocal run_called
                run_called = True

            def analyze(self):
                nonlocal analyze_called
                analyze_called = True

        self.assertEqual(_TestExperiment.__doc__, '_TestExperiment')

        args = (1, 2)
        kwargs = {'foo': 3, 'bar': 4}

        with dax.util.artiq.get_managers() as managers:
            # Build the experiment
            exp = _TestExperiment(managers, *args, **kwargs)
            self.assertFalse(build_called)
            self.assertIsNone(build_args)
            self.assertIsNone(build_kwargs)
            self.assertFalse(prepare_called)
            self.assertFalse(run_called)
            self.assertFalse(analyze_called)

            # Prepare
            exp.prepare()
            self.assertFalse(build_called)
            self.assertIsNone(build_args)
            self.assertIsNone(build_kwargs)
            self.assertFalse(prepare_called)
            self.assertFalse(run_called)
            self.assertFalse(analyze_called)

            # Run
            exp.run()
            self.assertTrue(build_called)
            self.assertTupleEqual(build_args, args)
            self.assertDictEqual(build_kwargs, kwargs)
            self.assertTrue(prepare_called)
            self.assertTrue(run_called)
            self.assertFalse(analyze_called)

            # Analyze
            exp.analyze()
            self.assertTrue(analyze_called)
