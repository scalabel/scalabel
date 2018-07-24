/* global Sat SatImage SatLabel */

/**
 * Class for each video labeling session/task, uses SatImage items
 * @param {SatLabel} LabelType: label instantiation type
 */
function SatVideo(LabelType) {
  let self = this;
  Sat.call(self, SatImage, LabelType);

  self.tracks = []; // may not need this
  // self.videoName = document.getElementById('video_name').innerHTML;
  self.frameRate = document.getElementById('frame_rate').innerHTML;
  self.frameCounter = document.getElementById('frame_counter');
  self.playButton = document.getElementById('play_button');
  self.playButtonIcon = document.getElementById('play_button_icon');
  self.slider = document.getElementById('video_slider');
  self.numFrames = self.slider.max;

  self.currentItem = self.items[0];
  self.currentItem.image.onload = function() {
    self.currentItem.redraw();};

  self.playing = false;
  self.playButton.onclick = function() {self.clickPlayPause();};

  self.slider.oninput = function() {self.moveSlider();};
}

SatVideo.prototype = Object.create(Sat.prototype);

SatVideo.prototype.newLabel = function(optionalAttributes) {
  let self = this;
  let labelId = self.newLabelId();
  let track = new Track(self, labelId, optionalAttributes);
  self.labelIdMap[track.id] = track;
  self.labels.push(track);
  self.tracks.push(track);
  for (let i = self.currentItem.index; i < self.items.length; i++) {
    let labelId = self.newLabelId();
    if (i > self.currentItem.index) {
      optionalAttributes.shadow = true;
      optionalAttributes.satItem = self.items[i];
      optionalAttributes.mousePos = null;
    }
    let childLabel = new self.LabelType(self, labelId, optionalAttributes);
    childLabel.parent = track;
    self.labelIdMap[childLabel.id] = childLabel;
    self.labels.push(childLabel);
    track.addChild(childLabel);
    self.items[i].labels.push(childLabel);
    childLabel.satItem = self.items[i];
  }
  return self.currentItem.labels[
    self.currentItem.labels.length - 1];
};

SatVideo.prototype.toJson = function() {
  let self = this;
  let json = self.encodeBaseJson();
  json.tracks = [];
  for (let i = 0; i < self.tracks.length; i++) {
    if (self.tracks[i].valid) {
      json.tracks.push(self.tracks[i].toJson());
    }
  }
  json.task.projectOptions.metadata = self.metadata;
  return json;
};

SatVideo.prototype.fromJson = function(json) {
  let self = this;
  self.decodeBaseJson(json);
  self.metadata = json.task.projectOptions.metadata;
  self.interpolationMode = json.task.projectOptions.interpolationMode;
  self.tracks = [];
  for (let i = 0; json.tracks && i < json.tracks.length; i++) {
    let track = new Track(self, json.tracks[i].id,
      json.tracks[i].attributes);
    track.children = [];
    for (let j = 0; j < json.tracks[i].childrenIds.length; j++) {
      track.addChild(self.labelIdMap[json.tracks[i].childrenIds[j]]);
      self.labelIdMap[json.tracks[i].childrenIds[j]].parent = track;
      for (let k = 0; k < self.labels.length; k++) {
        if (self.labels[k].id === track.id) {
          self.labels[k] = track;
        }
      }
    }
    self.labelIdMap[json.tracks[i].id] = track;
    self.tracks.push(track);
  }
};

SatVideo.prototype.gotoItem = function(index) {
  let self = this;
  if (index >= 0 && index < self.items.length) {
    index = (index + self.items.length) % self.items.length;
    self.currentItem.setActive(false);
    self.currentItem = self.items[index];
    self.frameCounter.innerHTML = self.currentItem.index + 1;
    self.currentItem.setActive(true);
    self.currentItem.onload = function() {
      self.currentItem.redraw();
    };
    self.currentItem.redraw();
    self.slider.value = index + 1;
    self.frameCounter.innerHTML = index + 1;
  }
};

SatVideo.prototype.clickPlayPause = function(e) {
  let self = this;

  if (e) {
    e.preventDefault();
  }

  // switch the play status
  self.playing = !self.playing;

  // update the icon and play/pause the vid
  if (self.playing) {
    self.playButtonIcon.className = 'fa fa-pause';
    self.intervalID = setInterval(function() {self.nextFrame();},
      1000/self.frameRate);
  } else {
    self.playButtonIcon.className = 'fa fa-play';
    clearInterval(self.intervalID);
  }
};

SatVideo.prototype.nextFrame = function() {
  let self = this;
  if (self.currentItem.index < self.numFrames - 1) {
    self.currentItem = self.items[self.currentItem.index + 1];
    self.slider.value = self.currentItem.index + 1;
    self.frameCounter.innerHTML = self.currentItem.index + 1;
    self.items[self.currentItem.index - 1].setActive(false);
    self.currentItem.setActive(true);
    self.currentItem.redraw();
  } else {
    self.clickPlayPause();
  }
};

SatVideo.prototype.moveSlider = function() {
  let self = this;
  let oldItem = self.currentItem;
  self.currentItem = self.items[parseInt(self.slider.value) - 1];
  if (oldItem) {
    oldItem.setActive(false);
  }
  self.currentItem.setActive(true);
  self.frameCounter.innerHTML = self.currentItem.index + 1;
};


/**
 * TODO
 * @param {Sat} sat: The labeling session
 * @param {number} id: label object identifier
 * @param {object} optionalAttributes: Optional attributes for the SatLabel.
 * @constructor
 */
function Track(sat, id, optionalAttributes = null) {
  let self = this;
  SatLabel.call(self, sat, id, optionalAttributes);
  switch (self.sat.interpolationMode) {
    case 'linear':
      self.interpolateHandler = self.linearInterpolate;
      break;
    case 'force':
      self.interpolateHandler = self.forceInterpolate;
      break;
    default:
      self.interpolateHandler = self.linearInterpolate;
      break;
  }
}

Track.prototype = Object.create(SatLabel.prototype);

Track.prototype.getActive = function() {
  return this.active;
};

Track.prototype.setActive = function(active) {
  this.active = active;
};

Track.prototype.childDeleted = function() {
  // TODO: update the behavior of track on child being too small
  // this.delete();
};

Track.prototype.endTrack = function(endLabel) {
  let self = this;
  let endIndex = null;
  for (let i = 0; i < self.children.length; i++) {
    if (endIndex) {
      self.children[i].delete();
    }
    if (!endIndex && self.children[i].id === endLabel.id) {
      endIndex = i;
    }
  }
  self.children = self.children.slice(0, endIndex+1);
};

Track.prototype.interpolate = function(startLabel) {
  let self = this;
  startLabel.keyframe = true;
  let startIndex = null;
  let priorKeyFrameIndex = 0;
  let nextKeyFrameIndex = null;
  // get the prior and next keyframe indices and the start index
  for (let i = 0; i < self.children.length; i++) {
    if (self.children[i].id === startLabel.id) {
      startIndex = i;
    } else if (startIndex && !nextKeyFrameIndex && self.children[i].keyframe) {
      nextKeyFrameIndex = i;
    }
    if (!startIndex && self.children[i].keyframe) {
      priorKeyFrameIndex = i;
    }
  }
  self.interpolateHandler(startLabel, startIndex, priorKeyFrameIndex,
    nextKeyFrameIndex);
};

Track.prototype.linearInterpolate = function(startLabel, startIndex,
                                             priorKeyFrameIndex,
                                             nextKeyFrameIndex) {
  let self = this;
  // interpolate between the beginning of the track and the starting label
  for (let i = priorKeyFrameIndex + 1; i < startIndex; i++) {
    let weight = (i - priorKeyFrameIndex) / (startIndex - priorKeyFrameIndex);
    self.children[i].interpolateHandler(self.children[priorKeyFrameIndex],
        startLabel, weight);
    self.children[i].attributes = startLabel.attributes;
  }
  if (nextKeyFrameIndex) {
    // if there is a later keyframe, interpolate
    for (let i = startIndex + 1; i < nextKeyFrameIndex; i++) {
      let weight = (i - startIndex) / (nextKeyFrameIndex - startIndex);
      self.children[i].interpolateHandler(startLabel,
          self.children[nextKeyFrameIndex], weight);
      self.children[i].attributes = startLabel.attributes;
    }
  } else {
    // otherwise, apply changes to remaining items
    for (let i = startIndex + 1; i < self.children.length; i++) {
      self.children[i].interpolateHandler(startLabel, startLabel, 0);
      self.children[i].attributes = startLabel.attributes;
    }
  }
};

Track.prototype.forceInterpolate = function(startLabel, startIndex,
                                            ignoredPriorKeyFrameIndex,
                                            nextKeyFrameIndex) {
  let self = this;
  if (nextKeyFrameIndex) {
    // if there is a later keyframe, only interpolate until there
    for (let i = startIndex + 1; i < nextKeyFrameIndex; i++) {
      self.children[i].shrink(startLabel);
      self.children[i].attributes = startLabel.attributes;
    }
  } else {
    // otherwise, apply changes to remaining items
    for (let i = startIndex + 1; i < self.children.length; i++) {
      self.children[i].shrink(startLabel);
      self.children[i].attributes = startLabel.attributes;
    }
  }
};
