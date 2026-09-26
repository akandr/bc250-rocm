#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Enter (or run a command in) the Fedora 44 copy from the running Fedora 43 system.
# Usage: f44chroot.sh up | down | run <command...>
R=/mnt/btrtop/root-f44
case "$1" in
up)
  sudo mkdir -p /mnt/btrtop
  mountpoint -q /mnt/btrtop || sudo mount -o subvolid=5 /dev/nvme0n1p3 /mnt/btrtop
  for m in proc sys dev dev/pts dev/shm; do mountpoint -q $R/$m || sudo mount --bind /$m $R/$m; done
  mountpoint -q $R/home || sudo mount --bind /home $R/home
  sudo cp -L /etc/resolv.conf $R/etc/resolv.conf.chroot && sudo mount --bind $R/etc/resolv.conf.chroot $R/etc/resolv.conf 2>/dev/null || true
  ;;
down)
  sudo umount $R/etc/resolv.conf 2>/dev/null
  for m in home dev/shm dev/pts dev sys proc; do mountpoint -q $R/$m && sudo umount $R/$m; done
  mountpoint -q /mnt/btrtop && sudo umount /mnt/btrtop
  ;;
run)
  shift
  sudo chroot $R /usr/bin/env -i HOME=/home/akandr PATH=/usr/bin:/usr/sbin LANG=C.UTF-8 /usr/sbin/runuser -u akandr -- "$@"
  ;;
root)
  shift
  sudo chroot $R /usr/bin/env -i HOME=/root PATH=/usr/bin:/usr/sbin LANG=C.UTF-8 "$@"
  ;;
esac
