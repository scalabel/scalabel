/* global Sat SatImage */

/**
 * Class for each video labeling session/task, uses SatImage items
 * @param {SatLabel} LabelType: label instantiation type
 */
function VideoSat(LabelType) {
  let self = this;
  Sat.call(self, SatImage, LabelType);

  self.videoName = document.getElementById('video_name').innerHTML;
  self.frameRate = document.getElementById('frame_rate').innerHTML;
  self.frameCounter = document.getElementById('frame_counter');
  self.playButton = document.getElementById('play_button');
  self.playButtonIcon = document.getElementById('play_button_icon');
  self.slider = document.getElementById('video_slider');
  self.numFrames = self.slider.max;

  // initialize all of the items (SatImage objects, one per frame)
  for (let i = 1; i <= self.numFrames; i++) {
    let frameString = i.toString();
    while (frameString.length < 7) {
      frameString = '0' + frameString;
    }
    self.newItem('./frames/' + self.videoName.slice(0, -4) + '/f-' +
      frameString + '.jpg');
  }

  self.currentItem = 0;
  self.items[self.currentItem].setActive(true);
  self.items[self.currentItem].image.onload = function() {
    self.items[self.currentItem].redraw();};

  self.playing = false;
  self.playButton.onclick = function() {self.clickPlayPause();};

  self.slider.oninput = function() {self.moveSlider();};
}

VideoSat.prototype = Object.create(Sat.prototype);

VideoSat.prototype.clickPlayPause = function(e) {
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

VideoSat.prototype.nextFrame = function() {
  let self = this;
  if (self.currentItem < self.numFrames - 1) {
    self.currentItem++;
    self.slider.value = self.currentItem + 1;
    self.frameCounter.innerHTML = self.currentItem + 1;
    self.items[self.currentItem].setActive(true);
    self.items[self.currentItem - 1].setActive(false);
    self.items[self.currentItem].redraw();
  } else {
    self.clickPlayPause();
  }
};

VideoSat.prototype.moveSlider = function() {
  let self = this;
  let oldItem = self.currentItem;
  self.currentItem = parseInt(self.slider.value) - 1;
  self.items[self.currentItem].setActive(true);
  self.items[oldItem].setActive(false);
  self.frameCounter.innerHTML = self.currentItem + 1;
};
