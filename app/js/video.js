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

  self.load();
  self.currentItem = self.items[0];
  self.currentItem.setActive(true);
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
  let previousLabelId = -1; // -1 means null
  for (let i = self.currentItem.index; i < self.items.length; i++) {
    let labelId = self.newLabelId();
    let childLabel = new self.LabelType(self, labelId, optionalAttributes);
    childLabel.parent = track;
    self.labelIdMap[childLabel.id] = childLabel;
    self.labels.push(childLabel);
    track.addChild(childLabel);
    self.items[i].labels.push(childLabel);
    if (previousLabelId > -1) {
      self.labelIdMap[previousLabelId].nextLabelId = childLabel.id;
    }
    childLabel.previousLabelId = previousLabelId;
    previousLabelId = childLabel.id;
  }
  if (previousLabelId > -1) {
    self.labelIdMap[previousLabelId].nextLabelId = -1;
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
  json.metadata = self.metadata;
  return json;
};

SatVideo.prototype.fromJson = function(json) {
  let self = this;
  self.decodeBaseJson(json);
  self.metadata = json.metadata;
  self.tracks = [];
  for (let i = 0; json.tracks && i < json.tracks.length; i++) {
    let track = new Track(self, json.tracks[i].id,
      json.tracks[i].attributeValues);
    track.children = [];
    for (let j = 0; j < json.tracks[i].children.length; j++) {
      track.addChild(self.labelIdMap[json.tracks[i].children[j]]);
      self.labelIdMap[json.tracks[i].children[j]].parent = track;
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
  SatLabel.call(this, sat, id, optionalAttributes);
}

Track.prototype = Object.create(SatLabel.prototype);

Track.prototype.childDeleted = function() {
  // TODO: update the behavior of track on child being too small
  this.delete();
};

Track.prototype.endTrack = function(endLabel) {
  let self = this;
  let endIndex = null;
  for (let i = 0; i < self.children.length; i++) {
    if (endIndex) {
      self.children[i].delete();
    }
    if (self.children[i].id === endLabel.id) {
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
  // interpolate between the beginning of the track and the starting label
  for (let i = priorKeyFrameIndex; i < startIndex; i++) {
    let weight = i / startIndex;
    self.children[i].weightedAvg(self.children[priorKeyFrameIndex], startLabel,
      weight);
  }
  if (nextKeyFrameIndex) {
    // if there is a later keyframe, interpolate
    for (let i = startIndex + 1; i < nextKeyFrameIndex; i++) {
      let weight = (i - startIndex) / (nextKeyFrameIndex - startIndex);
      self.children[i].weightedAvg(startLabel, self.children[nextKeyFrameIndex],
        weight);
    }
  } else {
    // otherwise, just apply changes to remaining items
    for (let i = startIndex; i < self.children.length; i++) {
      self.children[i].weightedAvg(startLabel, startLabel, 0);
    }
  }
};
