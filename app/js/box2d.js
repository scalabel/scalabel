/* global SatImage ImageLabel */
/* exported Box2d Box2dImage*/

/**
 * 2D box image labeling
 * @param {Sat} sat: task context
 * @param {int} index: index of this image in the task
 * @param {string} url: source of the image
 * @constructor
 */
function Box2dImage(sat, index, url) {
  SatImage.call(this, sat, index, url);
  // TODO(Wenqi): Add more methods and variables
}

Box2dImage.prototype = Object.create(SatImage.prototype);

/**
 * 2D box label
 * @param {Sat} sat: context
 * @param {int} id: label id
 */
function Box2d(sat, id) {
  ImageLabel.call(this, sat, id);
  // TODO(Wenqi): Add more methods and variables
}

Box2d.prototype = Object.create(ImageLabel.prototype);
