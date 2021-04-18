import unittest
import typing
import random
import argparse

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
            (self._undecorated_func, undecorated, 'Undecorated'),
            (self._kernel_func, kernel_, 'Kernel'),
            (self._rpc_func, rpc_, 'RPC'),
            (self._rpc_async_func, rpc_async, 'Async RPC'),
            (self._portable_func, portable_, 'Portable'),
            (self._host_only_func, host_only_, 'Host only'),
        ]
        for test_function, ref, test_name in function_list:
            self.assertEqual(fn(test_function, **kwargs), ref, f'{test_name} function wrongly classified by {name}')

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

    def test_cloned_dataset_manager(self):
        with dax.util.artiq.get_managers() as managers:
            clone = dax.util.artiq.ClonedDatasetManager(managers.dataset_mgr)
            self.assertIs(clone.ddb, managers.dataset_mgr.ddb)
            self.assertIsInstance(clone, artiq.master.worker_db.DatasetManager)

    def test_cloned_dataset_manager_non_recursive(self):
        with dax.util.artiq.get_managers() as managers:
            clone = dax.util.artiq.ClonedDatasetManager(managers.dataset_mgr)
            with self.assertRaises(TypeError, msg='Recursive clone did not raise'):
                dax.util.artiq.ClonedDatasetManager(clone)

    def test_cloned_dataset_manager_name(self):
        with dax.util.artiq.get_managers() as managers:
            name = 'foobar'

            clone = dax.util.artiq.ClonedDatasetManager(managers.dataset_mgr, name=name)
            clone_dict = getattr(managers.dataset_mgr, dax.util.artiq.ClonedDatasetManager._CLONE_DICT_KEY)
            self.assertEqual(len(clone_dict), 1, 'Unexpected number of clones in dict')
            registered_clone_key, registered_clone = clone_dict.popitem()
            self.assertEqual(registered_clone_key, name, 'Dataset manager clone name was not passed correctly')
            self.assertIs(registered_clone, clone)

    def test_cloned_dataset_manager_name_index(self):
        with dax.util.artiq.get_managers() as managers:
            name = 'foobar_{index}'

            clone = dax.util.artiq.ClonedDatasetManager(managers.dataset_mgr, name=name)
            clone_dict = getattr(managers.dataset_mgr, dax.util.artiq.ClonedDatasetManager._CLONE_DICT_KEY)
            self.assertEqual(len(clone_dict), 1, 'Unexpected number of clones in dict')
            registered_clone_key, registered_clone = clone_dict.popitem()
            self.assertEqual(registered_clone_key, name.format(index=0),
                             'Dataset manager clone name was not passed correctly')
            self.assertIs(registered_clone, clone)

    def test_cloned_dataset_manager_unique_name(self):
        with dax.util.artiq.get_managers() as managers:
            name = 'foobar'

            dax.util.artiq.ClonedDatasetManager(managers.dataset_mgr, name=name)
            with self.assertRaises(LookupError, msg='Non-unique name did not raise'):
                dax.util.artiq.ClonedDatasetManager(managers.dataset_mgr, name=name)

    def test_cloned_dataset_manager_dataset_db(self):
        with dax.util.artiq.get_managers() as managers:
            d = {}
            clone = dax.util.artiq.ClonedDatasetManager(managers.dataset_mgr, dataset_db=d)
            self.assertIsNot(clone.ddb, managers.dataset_mgr.ddb)
            self.assertIs(clone.ddb, d)
            self.assertIsInstance(clone, artiq.master.worker_db.DatasetManager)

    def test_clone_managers(self):
        with dax.util.artiq.get_managers() as managers:
            write_hdf5_fn = managers.dataset_mgr.write_hdf5

            with dax.util.artiq.clone_managers(managers) as cloned:
                self.assertIs(managers.device_mgr, cloned.device_mgr, 'Device manager was modified unintentionally')
                self.assertIsNot(managers.dataset_mgr, cloned.dataset_mgr, 'Dataset manager was not replaced')
                self.assertIsNot(managers.dataset_mgr.write_hdf5, write_hdf5_fn,
                                 'write_hdf5() function was not replaced')
                self.assertIsInstance(cloned.dataset_mgr, dax.util.artiq.ClonedDatasetManager)
                self.assertIsNot(managers.argument_mgr, cloned.argument_mgr, 'Argument manager was not replaced')
                self.assertFalse(cloned.argument_mgr.unprocessed_arguments, 'Arguments not decoupled')
                self.assertIsNot(managers.scheduler_defaults, cloned.scheduler_defaults,
                                 'Scheduler defaults were not replaced')

    def test_clone_managers_name(self):
        with dax.util.artiq.get_managers() as managers:
            name = 'foo'

            with dax.util.artiq.clone_managers(managers, name=name) as cloned:
                clone_dict = getattr(managers.dataset_mgr, dax.util.artiq.ClonedDatasetManager._CLONE_DICT_KEY)
                self.assertEqual(len(clone_dict), 1, 'Unexpected number of clones in dict')
                registered_clone_key, registered_clone = clone_dict.popitem()
                self.assertEqual(registered_clone_key, name, 'Dataset manager clone name was not passed correctly')
                self.assertIs(registered_clone, cloned.dataset_mgr)

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
                self.assertEqual(len(managers), len(isolated))
                for m, i in zip(managers, isolated):
                    if not isinstance(m, dict):
                        self.assertNotEqual(m, i)
                    else:
                        self.assertFalse(i)
                    self.assertIsNot(m, i)
                self.assertFalse(isolated.argument_mgr.unprocessed_arguments, 'Arguments not decoupled')

    def test_isolate_managers_name(self):
        with dax.util.artiq.get_managers() as managers:
            name = 'foo'
            with dax.util.artiq.isolate_managers(managers, name=name) as isolated:
                clone_dict = getattr(managers.dataset_mgr, dax.util.artiq.ClonedDatasetManager._CLONE_DICT_KEY)
                self.assertEqual(len(clone_dict), 1, 'Unexpected number of clones in dict')
                registered_clone_key, registered_clone = clone_dict.popitem()
                self.assertEqual(registered_clone_key, name, 'Dataset manager clone name was not passed correctly')
                self.assertIs(registered_clone, isolated[1])

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
                    with self.assertRaises(TypeError, msg='Wrong scheduler object did not raise'):
                        dax.util.artiq.pause_strict_priority(self_.core)

            # Create the main experiment
            exp = TestExperiment(managers)
            exp.run()

    """Functions used for tests"""

    def _undecorated_func(self):
        pass

    @rpc
    def _rpc_func(self):
        pass

    @rpc(flags={'async'})
    def _rpc_async_func(self):
        pass

    @portable
    def _portable_func(self):
        pass

    @kernel
    def _kernel_func(self):
        pass

    @host_only
    def _host_only_func(self):
        pass


if __name__ == '__main__':
    unittest.main()
