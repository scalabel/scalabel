const SatLabel = require('../sat');

let parent = {id: 0, valid: true};
let child1 = {id: 2, valid: true};
let child2 = {id: 3, valid: true};
let child3 = {id: 4, valid: true};
let child4 = {id: 5, valid: true};
let sat = {
            labelIdMap: {
                0: parent, 2: child1,
                3: child2, 4: child3,
                5: child4,
            },
        };
let label = new SatLabel(sat, 1);
label.name = 'ToyLabel';
label.attributes = {attr1: 1, attr2: 2};
label.parent = parent;
label.children = [child1, child2, child3, child4];
label.numChildren = label.children.length;

let json1 = label.toJson();
let label2 = new SatLabel(sat);
label2.fromJsonVariables(json1);
label2.fromJsonPointers(json1);
let json2 = label2.toJson();

test('fromJson and toJson should preserve all the info', () => {
  expect(JSON.stringify(json1)).toBe(JSON.stringify(json2));
});
