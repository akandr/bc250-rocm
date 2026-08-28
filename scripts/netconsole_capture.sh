#!/usr/bin/env bash
# Capture the kernel log of a machine that is about to stop writing to disk.
#
# The journal ends mid-sentence when this board hangs, which says nothing about
# why. Netconsole ships each line by UDP as it is produced, so the record
# survives the machine. This is how the GPU reset was found to succeed before the
# host stalls, a fact the journal could never have shown.
#
# Run the listener on the workstation, not the board:
#   nc -u -l 6969 > netconsole.log
#
# The files this produces contain NUL bytes, sometimes far more NUL than text:
# the largest capture kept here is 94K of which 1430 bytes is the 18 lines that
# matter. Why they are there has not been established, so it is recorded as an
# observation rather than explained. The consequence is worth knowing before you
# open one: git and GitHub treat them as binary and will not show a diff or a
# preview, and a plain `cat` prints little. Read them with
#   tr -d '\0' < netconsole.log
# which is also how every quotation from them in this repository was taken.
# then arm the board (adjust addresses, and use a port nothing else broadcasts
# on; 6666 collides with smart-plug discovery on this network):
set -u
LISTENER_IP=${LISTENER_IP:-192.168.3.120}
LISTENER_PORT=${LISTENER_PORT:-6969}
BOARD_IP=${BOARD_IP:-192.168.3.151}
BOARD_IF=${BOARD_IF:-enp4s0}
MAC=$(ip neigh show "$LISTENER_IP" | awk '{print $5}' | head -1)
[ -z "$MAC" ] && { echo "no ARP entry for $LISTENER_IP; ping it first"; exit 1; }
sudo rmmod netconsole 2>/dev/null || true
sudo modprobe netconsole \
  netconsole=$((LISTENER_PORT-1))@$BOARD_IP/$BOARD_IF,$LISTENER_PORT@$LISTENER_IP/$MAC
sudo dmesg -n 8
echo "netconsole armed" | sudo tee /dev/kmsg > /dev/null
echo "armed: $BOARD_IP -> $LISTENER_IP:$LISTENER_PORT via $MAC"
