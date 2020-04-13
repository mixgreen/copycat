# Device DB used for testing

core_addr = '0.0.0.0'

device_db = {
    # Core device
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': core_addr, 'ref_period': 1e-9}
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

    # Generic TTL
    'ttl0': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLInOut',
        'arguments': {'channel': 0},
        'comment': 'This is a fairly long comment, shown as tooltip.'
    },
    'ttl1': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLOut',
        'arguments': {'channel': 1},
        'comment': 'Hello World'
    },

    # Aliases
    'alias_0': 'ttl1',
    'alias_1': 'alias_0',
    'alias_2': 'alias_1',

    # Looped alias
    'loop_alias_0': 'loop_alias_0',
    'loop_alias_1': 'loop_alias_0',
    'loop_alias_2': 'loop_alias_4',
    'loop_alias_3': 'loop_alias_2',
    'loop_alias_4': 'loop_alias_3',

    # Dead aliases
    'dead_alias_0': 'this_key_does_not_exist_123',
    'dead_alias_1': 'dead_alias_0',
    'dead_alias_2': 'dead_alias_1',
}
