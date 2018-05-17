const Polygon = require('../seg2d');

let poly = new Polygon(257,[[0,0],[4,0],[4,4],[0,4]]);
let snapshot1 = JSON.stringify(poly.toJson());
poly.reverse();
poly.reverse();
let snapshot2 = JSON.stringify(poly.toJson());
let poly2 = Polygon.fromJson(poly.toJson());
let snapshot3 = JSON.stringify(poly2.toJson());

test('Double reverse should preserve structure', () => {
  expect(snapshot2).toBe(snapshot1);
});
test('to and from Json should preserve structure', () => {
  expect(snapshot3).toBe(snapshot1);
});