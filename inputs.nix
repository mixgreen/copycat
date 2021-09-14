{ pkgs ? import <nixpkgs> {}, artiqpkgs ? import <artiq-full> { inherit pkgs; } }:

ps: (
  (with ps; [ numpy scipy pyvcd natsort pygit2 matplotlib graphviz h5py networkx sortedcontainers ]) ++
  (with artiqpkgs; [ artiq sipyco ])
)
