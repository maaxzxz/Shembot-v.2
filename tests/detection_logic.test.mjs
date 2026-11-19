import assert from "node:assert/strict";
import {
  boxesAreDuplicate,
  consolidateDetections,
  isTrackAbsorbedByMergedDetection,
  selectVerifiedPeople,
} from "../detection_logic.mjs";

const person = { bbox: [100, 100, 220, 480], score: 0.86, verifiedByPose: true };
const torso = { bbox: [120, 110, 190, 230], score: 0.89, verifiedByPose: true };
const legs = { bbox: [145, 300, 145, 260], score: 0.68, verifiedByPose: true };

assert.equal(boxesAreDuplicate(person.bbox, torso.bbox), true);
assert.equal(boxesAreDuplicate(person.bbox, legs.bbox), true);
assert.equal(consolidateDetections([person, torso, legs]).length, 1);
assert.deepEqual(consolidateDetections([person, torso, legs])[0].bbox, person.bbox);

const leftPerson = { bbox: [40, 80, 140, 360], score: 0.77 };
const rightPerson = { bbox: [190, 70, 145, 370], score: 0.81 };
assert.equal(boxesAreDuplicate(leftPerson.bbox, rightPerson.bbox), false);
assert.equal(consolidateDetections([leftPerson, rightPerson]).length, 2);

const fullScan = { bbox: [70, 90, 170, 400], score: 0.72, fromFullScan: true };
const tileScan = { bbox: [75, 95, 165, 395], score: 0.91, fromFullScan: false };
assert.equal(consolidateDetections([fullScan, tileScan]).length, 1);

// Four overlapping tile scans can see the head/torso/legs of one person as
// separate boxes. A shared full-frame pose identity must collapse all four.
const tileFragments = [
  { bbox: [142, 0, 240, 286], score: 0.9, poseGroupId: "pose-1" },
  { bbox: [280, 8, 235, 273], score: 0.6, poseGroupId: "pose-1" },
  { bbox: [126, 207, 252, 289], score: 0.71, poseGroupId: "pose-1" },
  { bbox: [280, 207, 260, 289], score: 0.73, poseGroupId: "pose-1" },
];
const mergedTileFragments = consolidateDetections(tileFragments);
assert.equal(mergedTileFragments.length, 1);
assert.deepEqual(mergedTileFragments[0].bbox, [126, 0, 414, 496]);
assert.equal(mergedTileFragments[0].mergedByPose, true);
assert.equal(
  isTrackAbsorbedByMergedDetection(
    { bbox: [142, 0, 240, 286] },
    mergedTileFragments,
  ),
  true,
  "an old partial-body track must be removed once the pose union absorbs it",
);

const fakePose = {
  bbox: [400, 100, 180, 400],
  score: 0.99,
  poseGroupId: "pose-test",
};
assert.equal(
  selectVerifiedPeople([], [fakePose], 0.56, () => 5).length,
  0,
  "a pose without a detector result must never create a person",
);
assert.equal(
  selectVerifiedPeople(
    [{ bbox: [100, 100, 100, 260], score: 0.4 }],
    [fakePose],
    0.56,
    () => 2,
  ).length,
  1,
  "a weak detector result may be rescued by a matching human pose",
);
assert.equal(
  selectVerifiedPeople(
    [{ bbox: [100, 100, 100, 260], score: 0.4 }],
    [fakePose],
    0.56,
    () => 2,
  )[0].poseGroupId,
  "pose-test",
  "a detector result corresponding to a pose must inherit its identity",
);
assert.equal(
  selectVerifiedPeople(
    [{ bbox: [100, 100, 100, 260], score: 0.8 }],
    [],
    0.56,
    () => 0,
  ).length,
  1,
  "a strong detector result must work before the optional pose model loads",
);
assert.equal(
  selectVerifiedPeople(
    [{ bbox: [100, 100, 100, 260], score: 0.75 }],
    [],
    0.56,
    () => 0,
    { requirePose: true, exceptionalScore: 0.88 },
  ).length,
  0,
  "ordinary detector boxes must not bypass an available pose verifier",
);
assert.equal(
  selectVerifiedPeople(
    [{ bbox: [100, 100, 100, 260], score: 0.92 }],
    [],
    0.56,
    () => 0,
    { requirePose: true, exceptionalScore: 0.88 },
  ).length,
  1,
  "an exceptional detector result may survive a temporary pose miss",
);

console.log("detection logic tests passed");
