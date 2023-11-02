from experiments.repeated_cnots.correlations import (
    correlations,
    proba_outcome,
    proba_phase_flip,
    proba_two_phase_flips,
)

data = {
    '': {'p': 1},
    '+': {'p': 0.9568645959101777},
    '-': {'p': 0.043135404089821416},
    '++': {'p': 0.9568495477185163},
    '+-': {'p': 0.043150452281482905},
    '-+': {'p': 0.9564707781934418},
    '--': {'p': 0.04352922180655545},
    '+++': {'p': 0.9568478600517165},
    '++-': {'p': 0.043152139948281536},
    '+-+': {'p': 0.9564259249660604},
    '+--': {'p': 0.043574075033940465},
    '-++': {'p': 0.9567981912280507},
    '-+-': {'p': 0.0432018087719458},
    '--+': {'p': 0.9553439633219615},
    '---': {'p': 0.04465603667803925},
    '++++': {'p': 0.9568473560517464},
    '+++-': {'p': 0.04315264394825232},
    '++-+': {'p': 0.9564198909599},
    '++--': {'p': 0.04358010904010091},
    '+-++': {'p': 0.956792204385547},
    '+-+-': {'p': 0.04320779561445224},
    '+--+': {'p': 0.9552127064717764},
    '+---': {'p': 0.044787293528222284},
    '-+++': {'p': 0.9568407710540306},
    '-++-': {'p': 0.04315922894596727},
    '-+-+': {'p': 0.9562723681017424},
    '-+--': {'p': 0.04372763189825809},
    '--++': {'p': 0.9566456835059381},
    '--+-': {'p': 0.04335431649406104},
    '---+': {'p': 0.9521196221679427},
    '----': {'p': 0.04788037783205876},
    '+++++': {'p': 0.9568473126464535},
    '++++-': {'p': 0.043152687353545834},
    '+++-+': {'p': 0.9564190265588713},
    '+++--': {'p': 0.04358097344112943},
    '++-++': {'p': 0.956791290928951},
    '++-+-': {'p': 0.043208709071045646},
    '++--+': {'p': 0.9551949546932968},
    '++---': {'p': 0.044805045306702665},
    '+-+++': {'p': 0.9568398862275299},
    '+-++-': {'p': 0.04316011377247061},
    '+-+-+': {'p': 0.9562543428884988},
    '+-+--': {'p': 0.04374565711150001},
    '+--++': {'p': 0.9566278221871946},
    '+--+-': {'p': 0.04337217781280609},
    '+---+': {'p': 0.9517546665584143},
    '+----': {'p': 0.04824533344158564},
    '-++++': {'p': 0.956846448952426},
    '-+++-': {'p': 0.04315355104757253},
    '-++-+': {'p': 0.9563989446229693},
    '-++--': {'p': 0.04360105537703124},
    '-+-++': {'p': 0.9567714316307115},
    '-+-+-': {'p': 0.04322856836928498},
    '-+--+': {'p': 0.9547646809972336},
    '-+---': {'p': 0.045235319002766845},
    '--+++': {'p': 0.956820051567854},
    '--++-': {'p': 0.04317994843214437},
    '--+-+': {'p': 0.9558196125806263},
    '--+--': {'p': 0.04418038741937269},
    '---++': {'p': 0.9562070229747438},
    '---+-': {'p': 0.0437929770252539},
    '----+': {'p': 0.9437323947735528},
    '-----': {'p': 0.05626760522644879},
}


class TestClassCorrelations:
    def test_proba_outcome(self):
        assert proba_outcome('-', data) == data['-']['p']
        assert proba_outcome('+', data) == data['+']['p']
        assert proba_outcome('+-', data) == data['+']['p'] * data['+-']['p']
        assert (
            proba_outcome('+-+--', data)
            == data['+']['p']
            * data['+-']['p']
            * data['+-+']['p']
            * data['+-+-']['p']
            * data['+-+--']['p']
        )

    def test_proba_phase_flip(self):
        assert (
            proba_phase_flip(index=1, data=data, verbose=True) == data['-']['p']
        )
        assert proba_phase_flip(
            index=2, data=data, verbose=True
        ) == proba_outcome('+-', data) + proba_outcome('--', data)
        assert proba_phase_flip(
            index=3, data=data, verbose=True
        ) == proba_outcome('++-', data) + proba_outcome(
            '+--', data
        ) + proba_outcome(
            '-+-', data
        ) + proba_outcome(
            '---', data
        )

    def test_proba_two_phase_flips(self):
        assert proba_two_phase_flips(1, 2, data) == proba_outcome('--', data)
        assert proba_two_phase_flips(1, 3, data) == proba_outcome(
            '-+-', data
        ) + proba_outcome('---', data)
        assert proba_two_phase_flips(1, 4, data) == proba_outcome(
            '-++-', data
        ) + proba_outcome('-+--', data) + proba_outcome(
            '--+-', data
        ) + proba_outcome(
            '----', data
        )
        assert proba_two_phase_flips(1, 4, data) != proba_outcome(
            '-++-', data
        ) + proba_outcome('-+--', data) + proba_outcome(
            '--+-', data
        ) + proba_outcome(
            '---+', data
        )
        # wrong for the last outcome

    def test_correlations(self):
        assert correlations(1, 2, data) == proba_outcome(
            '--', data
        ) - proba_outcome('-', data) * (
            proba_outcome('+-', data) + proba_outcome('--', data)
        )

        assert correlations(1, 3, data) == proba_two_phase_flips(
            1, 3, data
        ) - proba_phase_flip(1, data) * proba_phase_flip(3, data)

        assert correlations(1, 4, data) == proba_two_phase_flips(
            1, 4, data
        ) - proba_phase_flip(1, data) * proba_phase_flip(4, data)

        assert correlations(2, 5, data) == proba_two_phase_flips(
            2, 5, data
        ) - proba_phase_flip(2, data) * proba_phase_flip(5, data)


if __name__ == "__main__":
    test = TestClassCorrelations()
    test.test_proba_outcome()
    test.test_proba_phase_flip()
    test.test_proba_two_phase_flips()
    test.test_correlations()
