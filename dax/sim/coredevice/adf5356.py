# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False
# mypy: check_untyped_defs = False
# mypy: warn_return_any = False

from numpy import int32, int64, floor, ceil

from artiq.language.core import kernel, portable, delay
from artiq.language.units import us, MHz
from artiq.language.types import TInt32, TInt64
from artiq.coredevice.adf5356_reg import (  # type: ignore[import]
    ADF5356_REG4_MUXOUT_UPDATE,
    ADF5356_REG6_RF_OUTPUT_A_POWER_UPDATE, ADF5356_REG6_RF_OUTPUT_A_POWER_GET, ADF5356_REG6_RF_OUTPUT_A_ENABLE,
    ADF5356_REG4_R_COUNTER_UPDATE,
    ADF5356_REG0_PRESCALER,
    ADF5356_REG0_INT_VALUE_UPDATE, ADF5356_REG1_MAIN_FRAC_VALUE_UPDATE, ADF5356_REG2_AUX_FRAC_LSB_VALUE_UPDATE,
    ADF5356_REG2_AUX_MOD_LSB_VALUE, ADF5356_REG4_REF_MODE, ADF5356_REG4_PD_POLARITY,
    ADF5356_REG2_AUX_MOD_LSB_VALUE_UPDATE, ADF5356_REG13_AUX_FRAC_MSB_VALUE_UPDATE,
    ADF5356_REG13_AUX_MOD_MSB_VALUE_UPDATE, ADF5356_REG6_RF_DIVIDER_SELECT_UPDATE,
    ADF5356_REG6_CP_BLEED_CURRENT_UPDATE, ADF5356_REG9_VCO_BAND_DIVISION_UPDATE,
    ADF5356_REG4_CURRENT_SETTING, ADF5356_REG4_MUX_LOGIC, ADF5356_REG4_MUXOUT, ADF5356_REG4_R_DOUBLER,
    ADF5356_REG4_R_DIVIDER, ADF5356_REG4_R_COUNTER, ADF5356_REG6_NEGATIVE_BLEED, ADF5356_REG6_CP_BLEED_CURRENT,
    ADF5356_REG6_FB_SELECT, ADF5356_REG6_MUTE_TILL_LD, ADF5356_REG6_RF_OUTPUT_A_POWER, ADF5356_REG7_LE_SYNC,
    ADF5356_REG7_FRAC_N_LD_PRECISION, ADF5356_REG9_SYNTH_LOCK_TIMEOUT, ADF5356_REG9_AUTOCAL_TIMEOUT,
    ADF5356_REG9_TIMEOUT, ADF5356_REG9_VCO_BAND_DIVISION, ADF5356_REG10_ADC_ENABLE, ADF5356_REG10_ADC_CLK_DIV,
    ADF5356_REG10_ADC_CONV,
    ADF5356_REG4_R_COUNTER_GET, ADF5356_REG4_R_DOUBLER_GET, ADF5356_REG4_R_DIVIDER_GET,
    ADF5356_REG0_INT_VALUE_GET,
    ADF5356_REG1_MAIN_FRAC_VALUE_GET,
    ADF5356_REG13_AUX_FRAC_MSB_VALUE_GET, ADF5356_REG2_AUX_FRAC_LSB_VALUE_GET,
    ADF5356_REG13_AUX_MOD_MSB_VALUE_GET, ADF5356_REG2_AUX_MOD_LSB_VALUE_GET,
    ADF5356_REG6_RF_DIVIDER_SELECT_GET, ADF5356_REG0_PRESCALER_GET,
    ADF5356_NUM_REGS
)
from artiq.coredevice.adf5356 import (  # type: ignore[import]
    ADF5356_MIN_VCO_FREQ, ADF5356_MAX_VCO_FREQ, ADF5356_MAX_FREQ_PFD, ADF5356_MODULUS1,
    calculate_pll
)

from dax.util.artiq_version import ARTIQ_MAJOR_VERSION
from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager


class ADF5356(DaxSimDevice):
    kernel_invariants = {"cpld", "sw", "channel", "core", "sysclk"}

    def __init__(
            self,
            dmgr,
            cpld_device,
            sw_device,
            channel,
            ref_doubler=False,
            ref_divider=False,
            **kwargs
    ):
        # Call super
        super(ADF5356, self).__init__(dmgr, **kwargs)

        # From ARTIQ code
        self.cpld = dmgr.get(cpld_device)
        self.sw = dmgr.get(sw_device)
        self.channel = channel

        self.ref_doubler = ref_doubler
        self.ref_divider = ref_divider
        self.sysclk = self.cpld.refclk
        assert 10 <= self.sysclk / 1e6 <= 600

        self._init_registers()

        # Register signals
        signal_manager = get_signal_manager()
        self._init = signal_manager.register(self, 'init', bool, size=1)
        self._enable = signal_manager.register(self, 'enable', bool, size=1)
        self._frequency = signal_manager.register(self, 'freq', float)
        self._output_power = signal_manager.register(self, 'power', int)

    @kernel
    def init(self, blind=False):
        # From ARTIQ code
        if not blind:
            # MUXOUT = VDD
            self.regs[4] = ADF5356_REG4_MUXOUT_UPDATE(self.regs[4], 1)
            self.sync()
            delay(1000 * us)
            delay(800 * us)

            # MUXOUT = DGND
            self.regs[4] = ADF5356_REG4_MUXOUT_UPDATE(self.regs[4], 2)
            self.sync()
            delay(1000 * us)
            delay(800 * us)

            # MUXOUT = digital lock-detect
            self.regs[4] = ADF5356_REG4_MUXOUT_UPDATE(self.regs[4], 6)
        else:
            self.sync()

        # Update signal
        self._init.push(True)

    if ARTIQ_MAJOR_VERSION >= 7:
        @kernel
        def set_att(self, att):
            self.cpld.set_att(self.channel, att)

    @kernel
    def set_att_mu(self, att):
        self.cpld.set_att_mu(self.channel, att)

    @kernel
    def write(self, data):
        raise NotImplementedError

    @kernel
    def read_muxout(self):
        raise NotImplementedError

    @kernel
    def set_output_power_mu(self, n):
        # From ARTIQ code
        if n not in [0, 1, 2, 3]:
            raise ValueError("invalid power setting")
        self.regs[6] = ADF5356_REG6_RF_OUTPUT_A_POWER_UPDATE(self.regs[6], n)
        self.sync()

        # Update signal
        self._output_power.push(n)

    @portable
    def output_power_mu(self):
        return ADF5356_REG6_RF_OUTPUT_A_POWER_GET(self.regs[6])

    @kernel
    def enable_output(self):
        # From ARTIQ code
        self.regs[6] |= ADF5356_REG6_RF_OUTPUT_A_ENABLE(1)
        self.sync()

        # Update signal
        self._enable.push(True)

    @kernel
    def disable_output(self):
        # From ARTIQ code
        self.regs[6] &= ~ADF5356_REG6_RF_OUTPUT_A_ENABLE(1)
        self.sync()

        # Update signal
        self._enable.push(False)

    @kernel
    def set_frequency(self, f):
        # From ARTIQ code
        freq = int64(round(f))

        if freq > ADF5356_MAX_VCO_FREQ:
            raise ValueError("Requested too high frequency")

        # select minimal output divider
        rf_div_sel = 0
        while freq < ADF5356_MIN_VCO_FREQ:
            freq <<= 1
            rf_div_sel += 1

        if (1 << rf_div_sel) > 64:
            raise ValueError("Requested too low frequency")

        # choose reference divider that maximizes PFD frequency
        self.regs[4] = ADF5356_REG4_R_COUNTER_UPDATE(
            self.regs[4], self._compute_reference_counter()
        )
        f_pfd = self.f_pfd()

        # choose prescaler
        if freq > int64(6e9):
            self.regs[0] |= ADF5356_REG0_PRESCALER(1)  # 8/9
            n_min, n_max = 75, 65535

            # adjust reference divider to be able to match n_min constraint
            while n_min * f_pfd > freq:
                r = ADF5356_REG4_R_COUNTER_GET(self.regs[4])
                self.regs[4] = ADF5356_REG4_R_COUNTER_UPDATE(self.regs[4], r + 1)
                f_pfd = self.f_pfd()
        else:
            self.regs[0] &= ~ADF5356_REG0_PRESCALER(1)  # 4/5
            n_min, n_max = 23, 32767

        # calculate PLL parameters
        n, frac1, (frac2_msb, frac2_lsb), (mod2_msb, mod2_lsb) = calculate_pll(
            freq, f_pfd
        )

        if not (n_min <= n <= n_max):
            raise ValueError("Invalid INT value")

        # configure PLL
        self.regs[0] = ADF5356_REG0_INT_VALUE_UPDATE(self.regs[0], n)
        self.regs[1] = ADF5356_REG1_MAIN_FRAC_VALUE_UPDATE(self.regs[1], frac1)
        self.regs[2] = ADF5356_REG2_AUX_FRAC_LSB_VALUE_UPDATE(self.regs[2], frac2_lsb)
        self.regs[2] = ADF5356_REG2_AUX_MOD_LSB_VALUE_UPDATE(self.regs[2], mod2_lsb)
        self.regs[13] = ADF5356_REG13_AUX_FRAC_MSB_VALUE_UPDATE(
            self.regs[13], frac2_msb
        )
        self.regs[13] = ADF5356_REG13_AUX_MOD_MSB_VALUE_UPDATE(self.regs[13], mod2_msb)

        self.regs[6] = ADF5356_REG6_RF_DIVIDER_SELECT_UPDATE(self.regs[6], rf_div_sel)
        self.regs[6] = ADF5356_REG6_CP_BLEED_CURRENT_UPDATE(
            self.regs[6], int32(floor(24 * f_pfd / (61.44 * MHz)))
        )
        self.regs[9] = ADF5356_REG9_VCO_BAND_DIVISION_UPDATE(
            self.regs[9], int32(ceil(f_pfd / 160e3))
        )

        # commit
        self.sync()

        # Update signal
        self._frequency.push(f)

    @kernel
    def sync(self):
        # From ARTIQ code
        f_pfd = self.f_pfd()
        delay(200 * us)  # Slack

        if f_pfd <= 75.0 * MHz:
            delay(200 * us)
        else:
            delay(200 * us)  # Slack
            delay(200 * us)

    @portable
    def f_pfd(self) -> TInt64:
        r = ADF5356_REG4_R_COUNTER_GET(self.regs[4])
        d = ADF5356_REG4_R_DOUBLER_GET(self.regs[4])
        t = ADF5356_REG4_R_DIVIDER_GET(self.regs[4])
        return self._compute_pfd_frequency(r, d, t)

    @portable
    def f_vco(self) -> TInt64:
        return int64(
            self.f_pfd() * (self.pll_n() + (self.pll_frac1() + self.pll_frac2() / self.pll_mod2()) / ADF5356_MODULUS1)
        )

    @portable
    def pll_n(self) -> TInt32:
        return ADF5356_REG0_INT_VALUE_GET(self.regs[0])

    @portable
    def pll_frac1(self) -> TInt32:
        return ADF5356_REG1_MAIN_FRAC_VALUE_GET(self.regs[1])

    @portable
    def pll_frac2(self) -> TInt32:
        return (ADF5356_REG13_AUX_FRAC_MSB_VALUE_GET(self.regs[13]) << 14) | ADF5356_REG2_AUX_FRAC_LSB_VALUE_GET(
            self.regs[2])

    @portable
    def pll_mod2(self) -> TInt32:
        return (ADF5356_REG13_AUX_MOD_MSB_VALUE_GET(self.regs[13]) << 14) | ADF5356_REG2_AUX_MOD_LSB_VALUE_GET(
            self.regs[2])

    @portable
    def ref_counter(self) -> TInt32:
        return ADF5356_REG4_R_COUNTER_GET(self.regs[4])

    @portable
    def output_divider(self) -> TInt32:
        return 1 << ADF5356_REG6_RF_DIVIDER_SELECT_GET(self.regs[6])

    def info(self):
        prescaler = ADF5356_REG0_PRESCALER_GET(self.regs[0])
        return {
            "f_outA": self.f_vco() / self.output_divider(),
            "f_outB": self.f_vco() * 2,
            "output_divider": self.output_divider(),

            "f_vco": self.f_vco(),
            "pll_n": self.pll_n(),
            "pll_frac1": self.pll_frac1(),
            "pll_frac2": self.pll_frac2(),
            "pll_mod2": self.pll_mod2(),
            "prescaler": "4/5" if prescaler == 0 else "8/9",

            "sysclk": self.sysclk,
            "ref_doubler": self.ref_doubler,
            "ref_divider": self.ref_divider,
            "ref_counter": self.ref_counter(),
            "f_pfd": self.f_pfd(),
        }

    @portable
    def _init_registers(self):
        self.regs = [int32(i) for i in range(ADF5356_NUM_REGS)]

        self.regs[2] |= ADF5356_REG2_AUX_MOD_LSB_VALUE(1)

        if self.sysclk <= 250 * MHz:
            self.regs[4] |= ADF5356_REG4_REF_MODE(0)
        else:
            self.regs[4] |= ADF5356_REG4_REF_MODE(1)

        self.regs[4] |= ADF5356_REG4_PD_POLARITY(1)

        self.regs[4] |= ADF5356_REG4_CURRENT_SETTING(2)

        self.regs[4] |= ADF5356_REG4_MUX_LOGIC(1)  # 3v3 logic
        self.regs[4] |= ADF5356_REG4_MUXOUT(6)

        if self.ref_doubler:
            self.regs[4] |= ADF5356_REG4_R_DOUBLER(1)

        if self.ref_divider:
            self.regs[4] |= ADF5356_REG4_R_DIVIDER(1)

        r = self._compute_reference_counter()
        self.regs[4] |= ADF5356_REG4_R_COUNTER(r)

        self.regs[5] = int32(0x800025)

        self.regs[6] = int32(0x14000006)

        self.regs[6] |= ADF5356_REG6_NEGATIVE_BLEED(1)

        self.regs[6] |= ADF5356_REG6_CP_BLEED_CURRENT(
            int32(floor(24 * self.f_pfd() / (61.44 * MHz)))
        )

        self.regs[6] |= ADF5356_REG6_FB_SELECT(1)

        self.regs[6] |= ADF5356_REG6_MUTE_TILL_LD(1)

        self.regs[6] |= ADF5356_REG6_RF_OUTPUT_A_ENABLE(1)

        self.regs[6] |= ADF5356_REG6_RF_OUTPUT_A_POWER(3)  # +5 dBm

        self.regs[7] = int32(0x10000007)

        self.regs[7] |= ADF5356_REG7_LE_SYNC(1)

        self.regs[7] |= ADF5356_REG7_FRAC_N_LD_PRECISION(3)

        self.regs[8] = int32(0x102D0428)

        self.regs[9] |= ADF5356_REG9_SYNTH_LOCK_TIMEOUT(13) | ADF5356_REG9_AUTOCAL_TIMEOUT(31) | ADF5356_REG9_TIMEOUT(
            0x67)

        self.regs[9] |= ADF5356_REG9_VCO_BAND_DIVISION(
            int32(ceil(self.f_pfd() / 160e3))
        )

        self.regs[10] = int32(0xC0000A)

        self.regs[10] |= (ADF5356_REG10_ADC_ENABLE(1) | ADF5356_REG10_ADC_CLK_DIV(256) | ADF5356_REG10_ADC_CONV(1))

        self.regs[11] = int32(0x61200B)

        self.regs[12] = int32(0x15FC)

    @portable
    def _compute_pfd_frequency(self, r, d, t) -> TInt64:
        return int64(self.sysclk * ((1 + d) / (r * (1 + t))))

    @portable
    def _compute_reference_counter(self) -> TInt32:
        d = ADF5356_REG4_R_DOUBLER_GET(self.regs[4])
        t = ADF5356_REG4_R_DIVIDER_GET(self.regs[4])
        r = 1
        while self._compute_pfd_frequency(r, d, t) > ADF5356_MAX_FREQ_PFD:
            r += 1
        return int32(r)
