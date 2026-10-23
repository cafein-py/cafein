#!/usr/bin/env bash
# Exit codes of small cafein programs on this runner, from a bare import up
# to the memory loop, to find where a non-zero exit at shutdown starts.
set +e
export PYTHONFAULTHANDLER=1
D=tests/data
run() {
  name="$1"; shift
  python "$@"
  echo "EXITCODE $name $?"
}
run import -c "import cafein"
run import_pyrosm -c "import cafein, pyrosm"
run build_gtfs -c "
from cafein import TransportNetwork
n = TransportNetwork.from_gtfs(['$D/helsinki_gtfs.zip'])
print('built', n)"
run build_osm -c "
from cafein import TransportNetwork
n = TransportNetwork.from_gtfs(['$D/helsinki_gtfs.zip'], osm_pbf='$D/kantakaupunki.osm.pbf')
print('built')"
run build_multimodal -c "
from cafein import TransportNetwork
n = TransportNetwork.from_gtfs(['$D/helsinki_gtfs.zip'], osm_pbf='$D/kantakaupunki.osm.pbf',
    street_modes=('walk', 'bicycle', 'e_scooter', 'wheelchair'))
print('built')"
run walking_read -c "
from cafein import streets
nodes, edges = streets._walking_network('$D/kantakaupunki.osm.pbf')
print(len(edges))"
run union_read -c "
from cafein import _osm
nodes, edges = _osm.union_network('$D/kantakaupunki.osm.pbf', car=True)
print(len(edges))"
run pyrosm_only -c "
import pyrosm
osm = pyrosm.OSM('$D/kantakaupunki.osm.pbf', engine='out_of_core', workers='auto')
nodes, edges = osm.get_network(network_type='walking', nodes=True)
print(len(edges))"
run pytest_small -m pytest -q -p no:cacheprovider tests/test_units.py
MEMPROF_DIR="$RUNNER_TEMP/memprof-small" run pytest_small_plugin -m pytest -q -p no:cacheprovider -p memprof_plugin -n 2 tests/test_units.py
run analyse_only -c "import pandas; print('pandas', pandas.__version__)"
run retention ci/memdiag/retention.py
run memloop ci/memdiag/memloop.py
