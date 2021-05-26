import torch
from torch import nn
from torch.nn import DataParallel
from attention_encoder import GraphAttentionEncoder, GCNEncoder
import math


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class Attention(nn.Module):
    def __init__(self, problem, embed_dim, n_encoder_layers, n_heads, node_dim=5, encoder_type='GAT',
                 tanh_clipping=10, throwback=1, decoding_type='local'):
        super(Attention, self).__init__()
        self.problem = problem
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping
        self.throwback = throwback
        self.decoding_type = decoding_type

        self.encoder = GraphAttentionEncoder(n_heads=n_heads, embed_dim=embed_dim, n_layers=n_encoder_layers,
                                             node_dim=node_dim) \
            if encoder_type == 'GAT' else GCNEncoder(embed_dim=embed_dim, node_dim=node_dim, n_layers=n_encoder_layers)

        if self.decoding_type == 'static':
            self.graph_embed_projection = nn.Linear(embed_dim, embed_dim, bias=True)

        else:
            self.graph_embed_projection = nn.Linear(embed_dim, embed_dim, bias=False)

        if self.decoding_type != 'static' and self.decoding_type != 'zero_context':
            self.prev_node_context_projection = nn.Linear(2 * throwback * embed_dim, embed_dim, bias=False)
            self.first_action_embedding = nn.Parameter(torch.Tensor(2 * throwback * self.embed_dim))
            self.first_action_embedding.data.uniform_(-1, 1)

        self.project_node_embeddings = nn.Linear(embed_dim, embed_dim, bias=False)

    # @profile(output_file='forward.prof', sort_by='cumulative', lines_to_print=20, strip_dirs=True)
    def forward(self, graph, n_nodes):

        batch_size = graph.num_graphs

        embeddings = self.encoder(graph).view(batch_size, n_nodes, -1)

        if self.decoding_type == 'local':
            log_p, pi, costs = self.decoding_local(graph, embeddings, batch_size, n_nodes)
        elif self.decoding_type == 'global':
            log_p, pi, costs = self.decoding_global(graph, embeddings, batch_size, n_nodes)
        else:
            log_p, pi, costs = self.decoding_static(graph, embeddings, batch_size, n_nodes)

        # get log_likelihood for corresponding action
        _log_p = log_p.gather(-1, pi.unsqueeze(-1)).squeeze(-1).sum(1)

        return costs, _log_p

    def select_node(self, probs):

        if self.decode_type == "greedy":
            _, selected = probs.max(1)

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def set_decode_type(self, decode_type):
        self.decode_type = decode_type

    def get_previous_node_context(self, embeddings, state):
        batch_size = embeddings.size()[0]
        if state.step == 0:
            return self.first_action_embedding.expand(batch_size, 1, 2 * self.throwback * self.embed_dim)
        else:
            last_action_embedding = embeddings.gather(1,
                                                      state.prev_a[:, None].expand(batch_size, 1, self.embed_dim))

            if self.problem.NAME == 'GC' or (self.problem.NAME == 'GP' and self.problem.state_version == 2):
                last_sol_embedding = torch.cat([state.sol_rep[k][state.sol_ind[k, 0, prev]] for k, prev
                                                in enumerate(state.prev_a)], dim=-1).view(batch_size, 1, self.embed_dim)
            else:
                last_sol_embedding = state.sol_rep

            prev_node_context = torch.cat((last_action_embedding, last_sol_embedding,
                                           *[state.context_throwback[i] for i in range(1, self.throwback)]), dim=-1)
            state.context_throwback.appendleft(torch.cat((last_action_embedding, last_sol_embedding), dim=-1))
            return prev_node_context

    # @profile(output_file='decoding6.prof', sort_by='cumulative', lines_to_print=20, strip_dirs=True)
    def decoding_local(self, graph, embeddings, batch_size, n_nodes):
        # get initial state
        state = self.problem.make_state(graph, embeddings, n_nodes, self.throwback, self.decoding_type)

        # initialize throwback context
        init_context = self.first_action_embedding.chunk(self.throwback, dim=-1)
        for a in init_context:
            state.context_throwback.append(a.expand(batch_size, 1, 2 * self.embed_dim))

        graph_embedding = torch.max(embeddings, 1)[0].view(batch_size, 1, self.embed_dim)
        graph_projection_fixed = self.graph_embed_projection(graph_embedding)
        attention_K = self.project_node_embeddings(embeddings)

        outputs = []
        sequences = []

        # decoding steps
        while not state.is_done():

            context_embedding = graph_projection_fixed + \
                                self.prev_node_context_projection(self.get_previous_node_context(embeddings, state))

            mask = state.visited > 0

            if state.step == 0:
                # if first iteration of decoder, compute attention coefficients to all nodes in graph
                logits = torch.matmul(context_embedding, attention_K.transpose(-2, -1)) / math.sqrt(
                    self.embed_dim)
                logits = (torch.tanh(logits) * self.tanh_clipping)

            else:
                # update the logits of the neighbors from the last selected node.
                v = torch.tensor(state.update_nodes, device=embeddings.device)

                if v.nelement() > 0:
                    graph_inds = v.detach() // n_nodes

                    update_contexts = context_embedding.squeeze(1).gather(0, graph_inds[:, None].expand(
                        len(v), self.embed_dim))
                    # get attention_k of nodes in v
                    v_attention_key = attention_K.view(batch_size * n_nodes, self.embed_dim)[v]

                    update_logits = torch.sum(update_contexts * v_attention_key, dim=-1) / math.sqrt(self.embed_dim)
                    update_logits = torch.tanh(update_logits) * self.tanh_clipping
                    logits = logits.view(-1).scatter(0, v, update_logits).view(batch_size, 1, n_nodes)

            logits = logits.masked_fill(mask, - math.inf)
            log_p = torch.log_softmax(logits, dim=-1)  # compute softmax over remaining nodes

            assert not torch.isnan(log_p).any()

            selected = self.select_node(log_p.exp()[:, 0, :])

            state.update(selected)

            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

        costs = state.get_cost()

        return torch.stack(outputs, 1), torch.stack(sequences, 1), costs

    def decoding_global(self, graph, embeddings, batch_size, n_nodes):
        # get initial state
        state = self.problem.make_state(graph, embeddings, n_nodes, self.throwback, self.decoding_type)

        # initialize throwback context
        init_context = self.first_action_embedding.chunk(self.throwback, dim=-1)
        for a in init_context:
            state.context_throwback.append(a.expand(batch_size, 1, 2 * self.embed_dim))

        graph_embedding = torch.max(embeddings, 1)[0].view(batch_size, 1, self.embed_dim)
        graph_projection_fixed = self.graph_embed_projection(graph_embedding)
        attention_K = self.project_node_embeddings(embeddings)

        outputs = []
        sequences = []

        # decoding steps
        while not state.is_done():

            context_embedding = graph_projection_fixed + \
                                self.prev_node_context_projection(self.get_previous_node_context(embeddings, state))

            mask = state.visited > 0

            logits = torch.matmul(context_embedding, attention_K.transpose(-2, -1)) / math.sqrt(
                self.embed_dim)
            logits = (torch.tanh(logits) * self.tanh_clipping)

            logits = logits.masked_fill(mask, - math.inf)
            log_p = torch.log_softmax(logits, dim=-1)  # compute softmax over remaining nodes

            assert not torch.isnan(log_p).any()

            selected = self.select_node(log_p.exp()[:, 0, :])

            state.update(selected)

            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

        costs = state.get_cost()

        return torch.stack(outputs, 1), torch.stack(sequences, 1), costs

    def decoding_static(self, graph, embeddings, batch_size, n_nodes):
        # get initial state
        state = self.problem.make_state(graph, embeddings, n_nodes, self.throwback, self.decoding_type)

        graph_embedding = torch.max(embeddings, 1)[0].view(batch_size, 1, self.embed_dim)
        graph_projection = self.graph_embed_projection(graph_embedding)
        attention_K = self.project_node_embeddings(embeddings)

        outputs = []
        sequences = []

        logits = torch.matmul(graph_projection, attention_K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        logits = (torch.tanh(logits) * self.tanh_clipping)

        # decoding steps
        while not state.is_done():
            mask = state.visited > 0
            logits = logits.masked_fill(mask, - math.inf)
            log_p = torch.log_softmax(logits, dim=-1)

            assert not torch.isnan(log_p).any()
            selected = self.select_node(log_p.exp()[:, 0, :])
            state.update(selected)

            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

        costs = state.get_cost()

        return torch.stack(outputs, 1), torch.stack(sequences, 1), costs

    def decoding_zero_context(self, graph, embeddings, batch_size, n_nodes):
        # get initial state
        state = self.problem.make_state(graph, embeddings, n_nodes, self.throwback, self.decoding_type)

        graph_embedding = torch.max(embeddings, 1)[0].view(batch_size, 1, self.embed_dim)
        graph_projection = self.graph_embed_projection(graph_embedding)
        attention_K = self.project_node_embeddings(embeddings)

        outputs = []
        sequences = []

        logits = torch.matmul(graph_projection, attention_K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        logits = (torch.tanh(logits) * self.tanh_clipping)

        # decoding steps
        while not state.is_done():
            mask = state.visited > 0
            logits = logits.masked_fill(mask, - math.inf)
            log_p = torch.log_softmax(logits, dim=-1)

            assert not torch.isnan(log_p).any()
            selected = self.select_node(log_p.exp()[:, 0, :])
            state.update(selected)

            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

        costs = state.get_cost()

        return torch.stack(outputs, 1), torch.stack(sequences, 1), costs
