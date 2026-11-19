export function boxArea(box) {
  return Math.max(0, box[2]) * Math.max(0, box[3]);
}

export function boxCenter(box) {
  return [box[0] + box[2] / 2, box[1] + box[3] / 2];
}

export function intersectionOverUnion(first, second) {
  const left = Math.max(first[0], second[0]);
  const top = Math.max(first[1], second[1]);
  const right = Math.min(first[0] + first[2], second[0] + second[2]);
  const bottom = Math.min(first[1] + first[3], second[1] + second[3]);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = boxArea(first) + boxArea(second) - intersection;
  return union > 0 ? intersection / union : 0;
}

export function intersectionOverSmaller(first, second) {
  const left = Math.max(first[0], second[0]);
  const top = Math.max(first[1], second[1]);
  const right = Math.min(first[0] + first[2], second[0] + second[2]);
  const bottom = Math.min(first[1] + first[3], second[1] + second[3]);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const smallerArea = Math.min(boxArea(first), boxArea(second));
  return smallerArea > 0 ? intersection / smallerArea : 0;
}

export function centerIsInside(innerBox, outerBox) {
  const [x, y] = boxCenter(innerBox);
  return (
    x >= outerBox[0] &&
    x <= outerBox[0] + outerBox[2] &&
    y >= outerBox[1] &&
    y <= outerBox[1] + outerBox[3]
  );
}

function horizontalOverlapOverSmaller(first, second) {
  const left = Math.max(first[0], second[0]);
  const right = Math.min(first[0] + first[2], second[0] + second[2]);
  return Math.max(0, right - left) / Math.max(1, Math.min(first[2], second[2]));
}

export function boxesAreDuplicate(first, second) {
  const overlap = intersectionOverUnion(first, second);
  if (overlap >= 0.38) return true;

  const firstArea = boxArea(first);
  const secondArea = boxArea(second);
  const smaller = firstArea <= secondArea ? first : second;
  const larger = smaller === first ? second : first;
  const containment = intersectionOverSmaller(first, second);
  const smallerCenter = boxCenter(smaller);
  const largerCenter = boxCenter(larger);
  const horizontalAlignment = horizontalOverlapOverSmaller(first, second);
  const centerXDistance = Math.abs(smallerCenter[0] - largerCenter[0]);

  // A tiled scan often returns the head, torso, or legs of the same person as
  // a smaller box inside the full-frame body box. Treat that as one person,
  // even when the two boxes have very different areas.
  if (
    containment >= 0.72 &&
    centerIsInside(smaller, larger) &&
    horizontalAlignment >= 0.55 &&
    centerXDistance <= Math.max(smaller[2] * 0.75, larger[2] * 0.36)
  ) {
    return true;
  }

  const areaRatio = Math.max(firstArea, secondArea) /
    Math.max(1, Math.min(firstArea, secondArea));
  if (areaRatio > 3.5) return false;

  const distance = Math.hypot(
    smallerCenter[0] - largerCenter[0],
    smallerCenter[1] - largerCenter[1],
  );
  const smallerDiagonal = Math.max(1, Math.hypot(smaller[2], smaller[3]));
  return distance < smallerDiagonal * 0.24;
}

function preferredDetection(first, second) {
  const firstArea = boxArea(first.bbox);
  const secondArea = boxArea(second.bbox);
  const containment = intersectionOverSmaller(first.bbox, second.bbox);
  const sameVerifiedPose =
    first.poseGroupId != null &&
    second.poseGroupId != null &&
    first.poseGroupId === second.poseGroupId;
  let preferred;

  if (sameVerifiedPose) {
    const left = Math.min(first.bbox[0], second.bbox[0]);
    const top = Math.min(first.bbox[1], second.bbox[1]);
    const right = Math.max(
      first.bbox[0] + first.bbox[2],
      second.bbox[0] + second.bbox[2],
    );
    const bottom = Math.max(
      first.bbox[1] + first.bbox[3],
      second.bbox[1] + second.bbox[3],
    );
    preferred = first.score >= second.score ? first : second;
    return {
      ...preferred,
      bbox: [left, top, right - left, bottom - top],
      score: Math.max(first.score, second.score),
      verifiedByPose: Boolean(first.verifiedByPose || second.verifiedByPose),
      fromFullScan: Boolean(first.fromFullScan || second.fromFullScan),
      poseGroupId: first.poseGroupId,
      mergedByPose: true,
    };
  }

  // Keep the body-sized box when a head/torso/leg box is nested inside it.
  if (containment >= 0.72 && Math.max(firstArea, secondArea) >=
      Math.min(firstArea, secondArea) * 1.35) {
    preferred = firstArea >= secondArea ? first : second;
  } else if (Boolean(first.verifiedByPose) !== Boolean(second.verifiedByPose)) {
    preferred = first.verifiedByPose ? first : second;
  } else {
    preferred = first.score >= second.score ? first : second;
  }

  return {
    ...preferred,
    bbox: [...preferred.bbox],
    score: Math.max(first.score, second.score),
    verifiedByPose: Boolean(first.verifiedByPose || second.verifiedByPose),
    fromFullScan: Boolean(first.fromFullScan || second.fromFullScan),
    poseGroupId: first.poseGroupId ?? second.poseGroupId ?? null,
  };
}

export function consolidateDetections(detections) {
  const countable = detections.filter((detection) => !detection.edgeCropped);
  const sorted = [...countable].sort((first, second) => {
    const verifiedDifference = Number(Boolean(second.verifiedByPose)) -
      Number(Boolean(first.verifiedByPose));
    if (verifiedDifference) return verifiedDifference;
    return second.score - first.score;
  });
  const unique = [];

  for (const detection of sorted) {
    const duplicateIndex = unique.findIndex((kept) => {
      const sameVerifiedPose =
        kept.poseGroupId != null &&
        detection.poseGroupId != null &&
        kept.poseGroupId === detection.poseGroupId;
      return sameVerifiedPose || boxesAreDuplicate(kept.bbox, detection.bbox);
    });
    if (duplicateIndex === -1) unique.push({ ...detection, bbox: [...detection.bbox] });
    else unique[duplicateIndex] = preferredDetection(unique[duplicateIndex], detection);
  }

  return unique;
}

export function selectVerifiedPeople(
  detections,
  poses,
  strongScore,
  matchQuality,
) {
  return detections.flatMap((detection) => {
    let bestPoseQuality = 0;
    let bestPose = null;
    for (const pose of poses) {
      const quality = matchQuality(detection, pose);
      if (quality > bestPoseQuality) {
        bestPoseQuality = quality;
        bestPose = pose;
      }
    }

    const verifiedByPose = bestPoseQuality >= 1.05;
    const correspondsToPose = bestPoseQuality >= 0.35;
    if (!verifiedByPose && detection.score < strongScore) return [];
    return [{
      ...detection,
      bbox: [...detection.bbox],
      verifiedByPose,
      poseGroupId: correspondsToPose ? bestPose?.poseGroupId ?? null : null,
    }];
  });
}

export function isTrackAbsorbedByMergedDetection(track, detections) {
  return detections.some((detection) =>
    detection.mergedByPose &&
    intersectionOverSmaller(track.bbox, detection.bbox) >= 0.72
  );
}
