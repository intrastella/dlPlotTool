![Alt text](./samples/images/layout1.png?raw=true "Title")

![Alt text](./samples/images/layout3.png?raw=true "Title")


# DL Plotting Tool 
A repo for creating easily all the necessary plots by just letting the 'DLPlotter' collect all the data. Everything else will be handled.
Writing useful plots can be very time-consuming let alone to present it in a proper way.
With this tool you can save your time and invest it in optimizing your model. If you have your own plots you want to add -
you can do this too.

**How to**:

- Collect the data for the Configuration Loss plot. (Hyperparameter configuration).
  Alternatively you can also just load a checkpoint.

        plotter = DLPlotter()
        model = MyModel()
        ...
        total_loss = 0
        for epoch in range(5):
            for step, (x, y) in enumerate(loader):
                ...
                output = model(x)
                loss = loss_func(output, y)
                total_loss += loss.item()
                ...
        config = dict(lr=0.001, batch_size=64, ...)
        plotter.collect_parameter("exp001"", config, total_loss / (5 * len(loader))
        plotter.construct()

- Collect the data for the Loss plot. (Training / validation)

        plotter = DLPlotter()
        model = MyModel()
        ...
        for epoch in range(5):
            for step, (x, y) in enumerate(loader):
                ...
                output = model(x)
                loss = loss_func(output, y)

                plotter.collect_loss("exp001", len(loaders), epoch, step, loss.item(), "train")
                ...
        plotter.construct()

- Collect the data for the Weight Distribution plot.

        plotter = DLPlotter()
        model = MyModel()
        ...
        for epoch in range(5):
            for step, (x, y) in enumerate(loader):
                ...
                weights = dict(layer1=model.layer1.weight.detach().clone(),
                               layer2=model.layer2.weight.detach().clone(), ...)

                plotter.collect_weights("exp001", len(loader), epoch, step, weights)
                ...
        plotter.construct()


<h2> To Do: </h2>

> 1. Accuracy figures as a subclass of WindowFig needs to be done. You can find it in folder ./dlFigures/ . <br>
> 2. Finish / create test files - you find them in ./pToolTest/ .
> 3. Improve Callbacks - update mechanism. You find them in constructor.py in the class DashStruct.
> 4. Loading data from checkpoints (pickle files) in DLPlotter class.
> 5. Adding custom plots also in DLPlotter class.
> 6. (I am atm fixing 2 bugs with the figures in the section 'Learning Progress')
