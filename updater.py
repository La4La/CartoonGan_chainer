import chainer
import chainer.functions as F


class CartoonGAN(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis, self.vgg = kwargs.pop('models')
        self.w = kwargs.pop('w')
        super().__init__(*args, **kwargs)

    def update_core(self):
        opt_gen = self.get_optimizer('gen')

        def _update(optimizer, loss):
            optimizer.target.cleargrads()
            loss.backward()
            optimizer.update()

        photo = self.get_iterator('main').next()
        photo = self.converter(photo, self.device)

        image_gen = self.gen(photo)

        # gen content loss
        loss_content = F.mean_absolute_error(photo, image_gen)
        # loss_content = F.mean_absolute_error(self.vgg(photo), self.vgg(image_gen.array))
        gen_loss = self.w * loss_content

        if self.dis is not None:
            opt_dis = self.get_optimizer('dis')

            image_batch = self.get_iterator('illusts').next()
            image = [image for image, _ in image_batch]
            edge_smoothed = [edge_smoothed for _, edge_smoothed in image_batch]
            image = self.converter(image, self.device)
            edge_smoothed = self.converter(edge_smoothed, self.device)

            # update dis
            y_image_gen = self.dis(image_gen)
            y_image = self.dis(image)
            y_edge_smoothed = self.dis(edge_smoothed)

            # dis cartoon image loss
            loss_image = F.average(F.softplus(-y_image))

            # dis edge_smoothed image loss
            loss_edge = F.average(F.softplus(y_edge_smoothed))

            # dis image_gen loss
            loss_gen_image = F.average(F.softplus(y_image_gen))

            # dis loss
            dis_loss = loss_image + loss_edge + loss_gen_image

            # update dis
            _update(opt_dis, dis_loss)
            chainer.report({'loss': dis_loss}, self.dis)

            # gen image_gen loss
            loss_gen_adv = F.average(F.softplus(-y_image_gen))
            gen_loss += loss_gen_adv

        # update gen
        _update(opt_gen, gen_loss)
        chainer.report({'loss': gen_loss, 'content': loss_content}, self.gen)
