-- DUNE protocol Wireshark Lua dissector (auto-detecta default x ensemble)
-- Funciona nos dois modos E no Wireshark GUI (sem precisar de -X):
--   * default  : eth | DUNE(8) | IPv4 ...
--   * ensemble : eth | DUNE(8) | Votes_t(NUM_CLASSES+1) | IPv4 ...
-- Heurística: depois dos 8 bytes do DUNE, espia o próximo byte. Se o nibble
-- alto for 4 (início de IPv4, versão 4) => não há Votes (default). Caso
-- contrário => há Votes (contadores de voto são <= nº de switches, nibble 0),
-- então pula NUM_CLASSES+1 bytes antes do IPv4.
--
-- NUM_CLASSES deve bater com o nº de classes do modelo (len(clf.classes_)).
local NUM_CLASSES = 7  -- classes of the classes_filter of cluster_analysis/src/params.ini

-- Nome de Proto único (o dune.lua do sistema já usa "dune"); os CAMPOS continuam
-- "dune.*" pra o tshark/-e e o GUI mostrarem igual. Este registra por último no
-- ethertype 0xd00e, então é ele que decodifica (e cobre os dois modos).
local dune = Proto("dune_ensemble", "DUNE Protocol (auto default/ensemble)")

local class_type_vals = {
    [0] = "PL (Packet Level)",
    [1] = "FL (Flow Level)"
}
local orig_ethertype_vals = {
    [0x0800] = "IPv4",
    [0x86DD] = "IPv6",
    [0x8847] = "MPLS Unicast",
    [0x8848] = "MPLS Multicast"
}

-- Fields (Dune_h)
local f = dune.fields
f.orig_ethertype = ProtoField.uint16("dune.orig_ethertype", "Original EtherType", base.HEX, orig_ethertype_vals)
f.class_type     = ProtoField.uint8 ("dune.class_type",     "Class Type",       base.DEC, class_type_vals)
f.class          = ProtoField.uint8 ("dune.class",          "Class",            base.DEC)
f.collision      = ProtoField.uint8 ("dune.collision",      "Collision",        base.DEC)
f.model_id       = ProtoField.uint8 ("dune.model_id",       "Model ID",         base.DEC)
f.mpls_label     = ProtoField.uint16("dune.mpls_label",     "MPLS Label",       base.DEC)

-- Fields (Votes_t) -- só aparecem quando o modo ensemble é detectado
f.majority_class = ProtoField.uint8("dune.votes.majority_class", "Majority Class", base.DEC)
local vote_fields = {}
for i = 0, NUM_CLASSES - 1 do
    local pf = ProtoField.uint8("dune.votes.class_" .. i, "Votes Class " .. i, base.DEC)
    f["votes_class_" .. i] = pf
    vote_fields[i] = pf
end

local ETHERTYPE_DUNE = 0xd00e
local DUNE_HDR_LEN = 8
local VOTES_HDR_LEN = NUM_CLASSES + 1

function dune.dissector(tvb, pinfo, tree)
    if tvb:len() < DUNE_HDR_LEN then return 0 end
    pinfo.cols.protocol = "DUNE"

    local subtree = tree:add(dune, tvb(0, DUNE_HDR_LEN), "DUNE Header")
    local offset = 0
    subtree:add(f.orig_ethertype, tvb(offset,2)); offset = offset + 2
    subtree:add(f.class_type,     tvb(offset,1)); offset = offset + 1
    subtree:add(f.class,          tvb(offset,1)); offset = offset + 1
    subtree:add(f.collision,      tvb(offset,1)); offset = offset + 1
    subtree:add(f.model_id,       tvb(offset,1)); offset = offset + 1
    subtree:add(f.mpls_label,     tvb(offset,2)); offset = offset + 2

    -- auto-detecção: o byte logo após o DUNE é o começo do IPv4 (nibble 4)?
    local has_votes = false
    if tvb:len() > offset then
        if tvb(offset,1):bitfield(0,4) ~= 4 then
            has_votes = true
        end
    end

    if has_votes and tvb:len() >= offset + VOTES_HDR_LEN then
        local votes_tree = tree:add(dune, tvb(offset, VOTES_HDR_LEN), "DUNE Votes")
        for i = 0, NUM_CLASSES - 1 do
            votes_tree:add(vote_fields[i], tvb(offset,1)); offset = offset + 1
        end
        votes_tree:add(f.majority_class, tvb(offset,1)); offset = offset + 1
    end

    -- repassa pro dissector do EtherType original (IPv4)
    local orig_ethertype = tvb(0,2):uint()
    local next_tvb = tvb(offset):tvb()
    local next_dissector = DissectorTable.get("ethertype"):get_dissector(orig_ethertype)
    if next_dissector then
        next_dissector:call(next_tvb, pinfo, tree)
    end
end

DissectorTable.get("ethertype"):add(ETHERTYPE_DUNE, dune)
