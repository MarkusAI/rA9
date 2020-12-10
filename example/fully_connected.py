import rA9.nn as nn
from rA9.optim import SGD
import jax.random as random
import rA9.nn.functional as F
from rA9.autograd import Variable
from rA9.nn.modules import Module
from rA9.utils import PoissonEncoder
from rA9.utils.data import DataLoader
from rA9.datasets.mnist import MnistDataset, collate_fn



class SNN(Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(in_features=784,out_features=100)
        self.active1 = nn.LIF()
        self.fc2 = nn.Linear(in_features=100, out_features=10)
        self.output = nn.Output(out_features=10)
        
        

    def forward(self, x, time):
      x, intime = self.active1(self.fc1(x), time)
      return self.output(self.fc2(x), time)


model = SNN()
model.train()

PeDurx = 70
batch_size = 32
Pencoder = PoissonEncoder(duration=PeDurx)
optimizer = SGD(model.parameters(), lr=0.01)

train_loader = DataLoader(dataset=MnistDataset(training=True, flatten=True),
                          collate_fn=collate_fn, shuffle=True,
                          batch_size=batch_size)

test_loader = DataLoader(dataset=MnistDataset(training=False, flatten=True),
                         collate_fn=collate_fn, shuffle=False,
                         batch_size=batch_size)

for epoch in range(15):
    for i, (data, target) in enumerate(train_loader):
        target = Variable(target)
        for t in range(PeDurx):
            rnum = random.uniform(key=random.PRNGKey(0), shape=data.shape)
            uin = (jnp.abs(data) / 2 > rnum).astype('float32')
            q = jnp.multiply(uin, jnp.sign(data))
            output, time = model(Variable(q), t)
            print(str(t))

        loss = F.Spikeloss(output, target, time_step=time)
        loss.backward()  # calc gradients
        optimizer.step()  # update gradients
        print("Epoch:" + str(epoch) +"Time:"+ str(i) + "loss:" + str(loss.data))
