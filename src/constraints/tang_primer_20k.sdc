# Tang Primer 20K onboard oscillator: 27 MHz = 37.037 ns period.
create_clock -name clk -period 37.037 -waveform {0.000 18.518} [get_ports {clk}]
