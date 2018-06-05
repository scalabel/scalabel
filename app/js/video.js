/* global Sat SatImage ImageLabel */

/**
 * Class for each video labeling session/task, uses SatImage items
 * @param {SatLabel} LabelType: label instantiation type
 */
function SatVideo(LabelType) {
  let self = this;
  Sat.call(self, SatImage, LabelType);

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
  let label = new self.LabelType(self, labelId, optionalAttributes);
  self.labelIdMap[label.id] = label;
  self.labels.push(label);
  let previousLabelId = -1;
  for (let i = self.currentItem.index; i < self.items.length; i++) {
    let labelId = self.newLabelId();
    let childLabel = new self.LabelType(self, labelId, optionalAttributes);
    childLabel.parent = label;
    self.labelIdMap[childLabel.id] = childLabel;
    self.labels.push(childLabel);
    label.addChild(childLabel);
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
  json.metadata = self.metadata;
  return json;
};

SatVideo.prototype.fromJson = function(json) {
  let self = this;
  self.decodeBaseJson(json);
  self.metadata = json.metadata;
};

// TODO: this needs to be agnostic of label type!!!
// Right now it assumes SatImage is a Box2d
SatVideo.prototype.interpolate = function(startSatLabel) {
  let self = this;
  startSatLabel.keyframe = true;

  let priorKeyframe = null;
  let nextKeyframe = null;
  for (let i = 0; i < self.currentItem.index; i++) {
    if (startSatLabel.parent.children[i].keyframe) {
      priorKeyframe = i;
      break;
    }
  }
  for (let i = self.currentItem.index + 1;
    i < startSatLabel.parent.children.length; i++) {
    if (startSatLabel.parent.children[i].keyframe) {
      nextKeyframe = i;
      break;
    }
  }
  if (priorKeyframe !== null) {
    // if there's an earlier keyframe, interpolate
    for (let i = priorKeyframe; i < self.currentItem.index; i++) {
      let weight = i/(self.currentItem.index);
      // TODO: this is the part that makes too many assumptions (maybe)
      startSatLabel.parent.children[i].weightedAvg(
        startSatLabel.parent.children[priorKeyframe], startSatLabel, weight);
    }
  }
  if (nextKeyframe !== null) {
    // if there's a later keyframe, interpolate
    for (let i = self.currentItem.index + 1; i < nextKeyframe; i++) {
      let weight = (i - self.currentItem.index) /
        (nextKeyframe - self.currentItem.index);
      // TODO: this is the part that makes too many assumptions (maybe)
      startSatLabel.parent.children[i].weightAvg(startSatLabel,
        startSatLabel.parent.children[nextKeyframe], weight);
    }
  } else {
    // otherwise, just apply change to remaining items
    for (let i = self.currentItem.index + 1;
      i < startSatLabel.parent.children.length; i++) {
      startSatLabel.parent.children[i].weightedAvg(
        startSatLabel, startSatLabel, 0);
    }
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
  ImageLabel.call(this, sat, id, optionalAttributes);
}

Track.prototype = Object.create(Track.prototype);
