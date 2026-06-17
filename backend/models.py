from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
import os

DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATABASE_DIR, exist_ok=True)
DATABASE_PATH = os.path.join(DATABASE_DIR, 'pinn_solver.db')

engine = create_engine(f'sqlite:///{DATABASE_PATH}', echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class SolveRecord(Base):
    __tablename__ = 'solve_records'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=True)
    pde_type = Column(String(50), nullable=False)
    equation_latex = Column(Text, nullable=False)
    ic_expression = Column(Text, default='')
    bc_left_expression = Column(Text, default='')
    bc_right_expression = Column(Text, default='')
    domain_x_min = Column(Float, default=0.0)
    domain_x_max = Column(Float, default=1.0)
    domain_t_min = Column(Float, default=0.0)
    domain_t_max = Column(Float, default=1.0)
    params_json = Column(Text, default='{}')
    layers_json = Column(Text, default='[2, 64, 64, 64, 1]')
    n_collocation = Column(Integer, default=10000)
    n_initial = Column(Integer, default=200)
    n_boundary = Column(Integer, default=200)
    epochs = Column(Integer, default=5000)
    learning_rate = Column(Float, default=1e-3)
    final_loss = Column(Float, nullable=True)
    training_history_json = Column(Text, nullable=True)
    model_weights_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'pde_type': self.pde_type,
            'equation_latex': self.equation_latex,
            'ic_expression': self.ic_expression,
            'bc_left_expression': self.bc_left_expression,
            'bc_right_expression': self.bc_right_expression,
            'domain_x_min': self.domain_x_min,
            'domain_x_max': self.domain_x_max,
            'domain_t_min': self.domain_t_min,
            'domain_t_max': self.domain_t_max,
            'params': json.loads(self.params_json) if self.params_json else {},
            'layers': json.loads(self.layers_json) if self.layers_json else [2, 64, 64, 64, 1],
            'n_collocation': self.n_collocation,
            'n_initial': self.n_initial,
            'n_boundary': self.n_boundary,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'final_loss': self.final_loss,
            'training_history': json.loads(self.training_history_json) if self.training_history_json else [],
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


Base.metadata.create_all(engine)


def get_session():
    return SessionLocal()
