#ifndef DUNE_EGRESS_DEPARSER_P4
#define DUNE_EGRESS_DEPARSER_P4

#include "dune_headers.p4"

control DuneEgressDeparser(
    packet_out pkt,
    in Headers_t hdr
)
{
    apply {
       pkt.emit(hdr.ethernet);
       pkt.emit(hdr.dune);
#ifdef ENSEMBLE_MODE
       pkt.emit(hdr.votes);
#endif
       pkt.emit(hdr.ipv4);
       pkt.emit(hdr.ipv4_options);
       pkt.emit(hdr.tcp);
       pkt.emit(hdr.udp);
    }
}

#endif
