# Validation ledger

- Main repository advanced to 50d5bfe; all1,062 local report files preserved and hashes verified. Both upstream and local report copies retained under source_update_backup.
- Isolated native Elastic source94c7d55. All25native model files equal training revision2c78ff6;704simulator/text files equal epoch5.
- Download13files verified against pinned HFcf5bc4fd986bc6877d613c4adea7c3b293ae0075 and release SHA256SUMS.
- New adapter tests initially failed because the adapter did not exist; tests_red.log. Implementation plus40Elastic/adapter/upstream tests passed; tests_green.log.
- Five relevant legacy input/scoring/renderer contracts passed; eval_protocol_tests.log. An exploratory run of the entire old training-continuation test file had6missing-fixture/helper failures because those training scripts are not part of this Elastic checkout. Those unrelated training tests are excluded from the evaluation gate; protocol_tests.log preserves the original output.
- Two existing restart regression tests passed against the copied helper; restart_tests.log.
- All3native/graph gates passed:8real inputs, seeds42/43/44, forward and reverse order; exact zero output difference. Strict state load and bit-equal loaded tensors verified.
- No native model math changed. Adapter preserves14Djoint inputs/outputs and original CPUnoise/PILimagearithmetic; graphspeed is measured queryonly, not total simulator throughput.
