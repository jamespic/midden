use core::panic;
use std::{cmp::min, collections::HashMap, hash::Hash, vec::IntoIter};

use xxhash_rust::xxh3::Xxh3DefaultBuilder;

/// Callback interface for the iterative SCC traversal.
pub trait GraphSCCVisitor {
    type NodeT;
    type NodeIdT: Eq + Hash + Copy;
    type NodeAccT: Clone;
    type SCCAccT: Clone;
    type ErrorT;

    fn next_unvisited_node(
        &mut self,
        already_visited: impl FnMut(&Self::NodeIdT) -> bool,
    ) -> Result<Option<Self::NodeT>, Self::ErrorT>;
    fn get_node_id(&self, node: &Self::NodeT) -> Self::NodeIdT;
    fn get_node_acc(&self, node: &Self::NodeT) -> Self::NodeAccT;
    fn get_successors(&mut self, node: &Self::NodeT) -> Result<Vec<Self::NodeT>, Self::ErrorT>;
    fn accumulate_node_values(&self, v1: &mut Self::NodeAccT, v2: &Self::NodeAccT);
    fn accumulate_scc_values(&self, v1: &mut Self::SCCAccT, v2: &Self::SCCAccT);
    fn add_node_value_to_scc_value(
        &self,
        node_acc: &Self::NodeAccT,
        this_scc: usize,
        scc_acc: Option<&Self::SCCAccT>,
    ) -> Self::SCCAccT;
    fn emit_result(
        &mut self,
        node_id: &Self::NodeIdT,
        node_acc_this_scc: &Self::NodeAccT,
        scc_acc: &Self::SCCAccT,
    ) -> Result<(), Self::ErrorT>;
    fn _acc_scc_options(
        &self,
        mut scc_acc1: &mut Option<Self::SCCAccT>,
        scc_acc2: Option<&Self::SCCAccT>,
    ) {
        match (&mut scc_acc1, &scc_acc2) {
            (None, None) | (Some(..), None) => (),
            (None, Some(acc)) => *scc_acc1 = Some((*acc).clone()),
            (Some(acc1), Some(acc2)) => {
                self.accumulate_scc_values(acc1, acc2);
            }
        }
    }
}

struct BookkeepingEntry {
    index: usize,
    lowlink: usize,
    scc: Option<usize>,
    on_stack: bool,
}

struct TarjanState<V: GraphSCCVisitor> {
    bookkeeping: HashMap<V::NodeIdT, BookkeepingEntry, Xxh3DefaultBuilder>,
    call_stack: Vec<CallStackFrame<V>>,
    stack: Vec<V::NodeIdT>,
    index: usize,
    next_scc_index: usize,
    scc_accs: HashMap<usize, V::SCCAccT, Xxh3DefaultBuilder>,
}

enum CallStackFrameState<V: GraphSCCVisitor> {
    NextIteration,
    ChildWait,
    ChildReturn {
        child_node_acc: Option<V::NodeAccT>,
        child_scc_acc: Option<V::SCCAccT>,
    },
    AfterIteration,
}

struct CallStackFrame<V: GraphSCCVisitor> {
    state: CallStackFrameState<V>,
    obj_id: V::NodeIdT,
    node_acc: V::NodeAccT,
    scc_acc: Option<V::SCCAccT>,
    successor_iter: IntoIter<V::NodeT>,
    current_successor_id: Option<V::NodeIdT>,
}

impl<V: GraphSCCVisitor> TarjanState<V> {
    fn push_new_frame(&mut self, visitor: &mut V, node: V::NodeT) -> Result<(), V::ErrorT> {
        let node_index = self.index;
        self.index += 1;
        let node_id = visitor.get_node_id(&node);
        let entry = BookkeepingEntry {
            index: node_index,
            lowlink: node_index,
            scc: None,
            on_stack: true,
        };
        self.bookkeeping.insert(node_id, entry);
        let node_acc = visitor.get_node_acc(&node);
        let successors = visitor.get_successors(&node)?;
        let frame = CallStackFrame {
            state: CallStackFrameState::NextIteration,
            obj_id: node_id,
            node_acc,
            scc_acc: None,
            successor_iter: successors.into_iter(),
            current_successor_id: None,
        };
        self.stack.push(frame.obj_id);
        self.call_stack.push(frame);
        Ok(())
    }
}

pub fn visit_sccs<V: GraphSCCVisitor>(visitor: &mut V) -> Result<(), V::ErrorT> {
    let mut state: TarjanState<V> = TarjanState {
        bookkeeping: HashMap::with_hasher(Xxh3DefaultBuilder::new()),
        call_stack: Vec::new(),
        stack: Vec::new(),
        index: 0,
        next_scc_index: 0,
        scc_accs: HashMap::with_hasher(Xxh3DefaultBuilder::new()),
    };

    loop {
        let call_stack_head = state.call_stack.pop();
        if let Some(mut frame) = call_stack_head {
            match frame.state {
                CallStackFrameState::NextIteration => {
                    if let Some(successor) = frame.successor_iter.next() {
                        let successor_id = visitor.get_node_id(&successor);
                        frame.current_successor_id = Some(successor_id);
                        if !state.bookkeeping.contains_key(&successor_id) {
                            frame.state = CallStackFrameState::ChildWait;
                            state.call_stack.push(frame);
                            state.push_new_frame(visitor, successor)?;
                        } else {
                            let successor_entry = &state.bookkeeping[&successor_id];
                            if successor_entry.on_stack {
                                let new_lowlink = min(
                                    state.bookkeeping[&frame.obj_id].lowlink,
                                    successor_entry.index,
                                );
                                state.bookkeeping.get_mut(&frame.obj_id).unwrap().lowlink =
                                    new_lowlink;
                            } else {
                                let linked_scc = successor_entry.scc.unwrap();
                                let linked_scc_acc = &state.scc_accs[&linked_scc];
                                visitor._acc_scc_options(&mut frame.scc_acc, Some(linked_scc_acc));
                            }
                            state.call_stack.push(frame);
                        }
                    } else {
                        frame.state = CallStackFrameState::AfterIteration;
                        state.call_stack.push(frame);
                    }
                }
                CallStackFrameState::ChildWait => panic!(
                    "We should not be returning to a frame waiting for a child without having processed the child"
                ),
                CallStackFrameState::ChildReturn {
                    ref child_node_acc,
                    ref child_scc_acc,
                } => {
                    if let Some(child_node_acc) = child_node_acc {
                        visitor.accumulate_node_values(&mut frame.node_acc, &child_node_acc);
                    }
                    visitor._acc_scc_options(&mut frame.scc_acc, child_scc_acc.as_ref());

                    let successor_id = frame.current_successor_id.unwrap();
                    let successor_lowlink = state.bookkeeping[&successor_id].lowlink;
                    let current_lowlink = state.bookkeeping[&frame.obj_id].lowlink;
                    state.bookkeeping.get_mut(&frame.obj_id).unwrap().lowlink =
                        min(current_lowlink, successor_lowlink);
                    frame.state = CallStackFrameState::NextIteration;
                    state.call_stack.push(frame);
                }
                CallStackFrameState::AfterIteration => {
                    let returned_node_acc = if state.bookkeeping[&frame.obj_id].index
                        == state.bookkeeping[&frame.obj_id].lowlink
                    {
                        let scc = state.next_scc_index;
                        state.next_scc_index += 1;

                        let mut scc_members: Vec<V::NodeIdT> = Vec::new();
                        loop {
                            let member = state.stack.pop().unwrap();
                            scc_members.push(member);
                            let bookkeeping_item = state.bookkeeping.get_mut(&member).unwrap();
                            bookkeeping_item.on_stack = false;
                            bookkeeping_item.scc = Some(scc);

                            if member == frame.obj_id {
                                break;
                            }
                        }
                        frame.scc_acc = Some(visitor.add_node_value_to_scc_value(
                            &frame.node_acc,
                            scc,
                            frame.scc_acc.as_ref(),
                        ));
                        state.scc_accs.insert(scc, frame.scc_acc.clone().unwrap());

                        for member_id in scc_members {
                            visitor.emit_result(
                                &member_id,
                                &frame.node_acc,
                                &frame.scc_acc.as_ref().unwrap(),
                            )?;
                        }
                        None
                        //     return node_acc, scc_acc
                    } else {
                        Some(frame.node_acc)
                    };

                    if let Some(parent_frame) = state.call_stack.last_mut() {
                        parent_frame.state = CallStackFrameState::ChildReturn {
                            child_node_acc: returned_node_acc,
                            child_scc_acc: frame.scc_acc,
                        };
                    }
                }
            }
        } else {
            let next_node = visitor.next_unvisited_node(|id| state.bookkeeping.contains_key(id))?;
            if let Some(node) = next_node {
                // Handle dodgy iterator implementations that might return the same node multiple times
                if !state.bookkeeping.contains_key(&visitor.get_node_id(&node)) {
                    state.push_new_frame(visitor, node)?; // Handle dodgy iterator implementations that might return the same node multiple times
                }
            } else {
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashSet, hash_map::Iter as HashMapIter},
        convert::Infallible,
    };

    use super::*;

    struct TestVisitor<'a> {
        graph: &'a HashMap<u64, Vec<u64>>,
        results: HashMap<u64, (HashSet<u64>, HashSet<u64>)>,
        iterator: HashMapIter<'a, u64, Vec<u64>>,
    }

    impl<'a> TestVisitor<'a> {
        fn new(graph: &'a HashMap<u64, Vec<u64>>) -> Self {
            let iterator = graph.iter();
            Self {
                graph,
                results: HashMap::new(),
                iterator,
            }
        }
    }

    impl<'a> GraphSCCVisitor for TestVisitor<'a> {
        type NodeT = u64;
        type NodeIdT = u64;
        type NodeAccT = HashSet<u64>;
        type SCCAccT = HashSet<u64>;
        type ErrorT = Infallible;

        fn next_unvisited_node(
            &mut self,
            mut already_visited: impl FnMut(&Self::NodeIdT) -> bool,
        ) -> Result<Option<Self::NodeT>, Self::ErrorT> {
            let mut result = None;
            for (node_id, _) in self.iterator.by_ref() {
                if !already_visited(node_id) {
                    result = Some(*node_id);
                    break;
                }
            }
            Ok(result)
        }

        fn get_node_id(&self, node: &Self::NodeT) -> Self::NodeIdT {
            *node
        }

        fn get_node_acc(&self, node: &Self::NodeT) -> Self::NodeAccT {
            HashSet::from([*node])
        }

        fn get_successors(&mut self, node: &Self::NodeT) -> Result<Vec<Self::NodeT>, Self::ErrorT> {
            Ok(self.graph[&node].clone())
        }

        fn accumulate_node_values(&self, v1: &mut Self::NodeAccT, v2: &Self::NodeAccT) {
            v1.extend(v2.iter().cloned());
        }

        fn accumulate_scc_values(&self, v1: &mut Self::SCCAccT, v2: &Self::SCCAccT) {
            v1.extend(v2.iter().cloned());
        }

        fn add_node_value_to_scc_value(
            &self,
            node_acc: &Self::NodeAccT,
            _this_scc: usize,
            scc_acc: Option<&Self::SCCAccT>,
        ) -> Self::SCCAccT {
            if let Some(scc_acc) = scc_acc {
                scc_acc.iter().chain(node_acc.iter()).cloned().collect()
            } else {
                node_acc.clone()
            }
        }

        fn emit_result(
            &mut self,
            node_id: &Self::NodeIdT,
            node_acc_this_scc: &Self::NodeAccT,
            scc_acc: &Self::SCCAccT,
        ) -> Result<(), Self::ErrorT> {
            self.results
                .insert(*node_id, (node_acc_this_scc.clone(), scc_acc.clone()));
            Ok(())
        }
    }

    #[test]
    fn test_tarjan() {
        // Example from Wikipedia page on Tarjan's algorithm

        let test_graph: HashMap<u64, Vec<u64>> = [
            (1, vec![2]),
            (2, vec![3]),
            (3, vec![1]),
            (4, vec![2, 5]),
            (5, vec![4, 6]),
            (6, vec![3, 7]),
            (7, vec![6]),
            (8, vec![5, 7, 8]),
        ]
        .iter()
        .cloned()
        .collect();
        let mut visitor = TestVisitor::new(&test_graph);
        visit_sccs(&mut visitor).unwrap();
        assert_eq!(
            visitor.results,
            [
                (1, (HashSet::from([1, 2, 3]), HashSet::from([1, 2, 3]))),
                (2, (HashSet::from([1, 2, 3]), HashSet::from([1, 2, 3]))),
                (3, (HashSet::from([1, 2, 3]), HashSet::from([1, 2, 3]))),
                (
                    4,
                    (HashSet::from([4, 5]), HashSet::from([1, 2, 3, 4, 5, 6, 7]))
                ),
                (
                    5,
                    (HashSet::from([4, 5]), HashSet::from([1, 2, 3, 4, 5, 6, 7]))
                ),
                (6, (HashSet::from([6, 7]), HashSet::from([1, 2, 3, 6, 7]))),
                (7, (HashSet::from([6, 7]), HashSet::from([1, 2, 3, 6, 7]))),
                (
                    8,
                    (HashSet::from([8]), HashSet::from([1, 2, 3, 4, 5, 6, 7, 8]))
                ),
            ]
            .iter()
            .cloned()
            .collect()
        );
    }
}
