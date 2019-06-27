import * as action from '../../action/creators';
import * as box2d from '../../action/box2d';
import { testJson } from '../test_objects';
import Session from '../../common/session';
import * as labels from '../../common/label_types';
import * as _ from 'lodash';
import { RectType } from '../../functional/types';

test('Add, change and delete box2d labels', () => {
  Session.devMode = false;
  Session.initStore(testJson);
  Session.dispatch(action.goToItem(0));
  Session.dispatch(box2d.addBox2dLabel([0], 1, 2, 3, 4));
  Session.dispatch(box2d.addBox2dLabel([0], 1, 2, 3, 4));
  Session.dispatch(box2d.addBox2dLabel([0], 1, 2, 3, 4));
  let state = Session.getState();
  expect(_.size(state.items[0].labels)).toBe(3);
  expect(_.size(state.items[0].shapes)).toBe(3);
  const labelIds: number[] = _.map(state.items[0].labels, (l) => l.id);
  let label = state.items[0].labels[labelIds[0]];
  expect(label.item).toBe(0);
  expect(label.type).toBe(labels.BOX_2D);
  let shape = state.items[0].shapes[label.shapes[0]] as RectType;
  // Check label ids
  _.forEach(state.items[0].labels, (v, i) => {
    expect(v.id).toBe(parseInt(i, 10));
  });
  // Check shape ids
  _.forEach(state.items[0].shapes, (v, i) => {
    expect(v.id).toBe(parseInt(i, 10));
  });
  expect(shape.x).toBe(1);
  expect(shape.y).toBe(2);
  expect(shape.w).toBe(3);
  expect(shape.h).toBe(4);

  Session.dispatch(action.changeLabelShape(shape.id, { x: 2, w: 5 }));
  state = Session.getState();
  label = state.items[0].labels[label.id];
  shape = state.items[0].shapes[label.shapes[0]] as RectType;
  // console.log(label, shape, state.items[0].shapes);
  expect(shape.x).toBe(2);
  expect(shape.y).toBe(2);
  expect(shape.w).toBe(5);
  expect(shape.h).toBe(4);

  Session.dispatch(action.deleteLabel(label.id));
  state = Session.getState();
  expect(_.size(state.items[0].labels)).toBe(2);
  expect(_.size(state.items[0].shapes)).toBe(2);
});
