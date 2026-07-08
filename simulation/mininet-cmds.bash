
sudo \
env \
PATH="~/p4/p4dev-python-venv/bin:$PATH" \
mn \
--custom=./dune/__init__.py,./dune/p4/node.py,./dune/p4/link.py,./dune/custom_topo.py \
--topo=dunefattree,models=configs/models/ton.json,models_dir=./models,objects_dir=./p4objects,log_dir=./logs,pcap_dir=./pcaps,test_pcap=./data/ToN_IoT_test.pcap,test_pcap_dir=./utils/experiment_pcaps,test_pps=100,pkt_num=1000,super_spines=1,pods=1,spines=1,leafs=1,hosts_per_leaf=1 \
--switch=p4simpleswitchgrpc,log_level=DEBUG \
--controller=p4controller,topo_class=dunefattree,topo=configs/topos/topo_ton.json,log_dir=./logs,log_level=INFO \
--link=p4link \
--arp

sudo \
env \
PATH="~/p4/p4dev-python-venv/bin:$PATH" \
mn \
--custom=./dune/__init__.py,./dune/p4/node.py,./dune/p4/link.py,./dune/custom_topo.py \
--topo=dunesimple,models=configs/models/ton_1st_model_only.json,models_dir=./models,objects_dir=./p4objects,log_dir=./logs,pcap_dir=./pcaps \
--switch=p4simpleswitchgrpc,log_level=DEBUG \
--controller=p4controller,topo_class=dunesimple,topo=configs/topos/simpletopo.json,log_dir=./logs,log_level=INFO \
--link=p4link \
--arp

editcap -T ether -L -C 12:2 ./data/ToN_IoT_test.pcap ./data/ToN_IoT_clean_ether.pcap
